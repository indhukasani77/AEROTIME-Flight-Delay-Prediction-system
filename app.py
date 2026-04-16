import os
import sys
import hashlib
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS

# ── PATHS ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR   = os.path.join(os.path.dirname(BASE_DIR), 'database')
sys.path.insert(0, DB_DIR)

from setup_db import get_connection, init_db

app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.secret_key = 'aerotime_secret_key_2026'
CORS(app, supports_credentials=True)

init_db()

# ════════════════════════════════════════════════
#  LOAD ML MODELS AT STARTUP
# ════════════════════════════════════════════════
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
ML_READY   = False
best_model = None
encoders   = None
feat_cols  = None
scaler     = None

# Real stats hardcoded from your model_meta.pkl output
MODEL_STATS = {
    "ml_ready":      False,
    "best_model":    "Random Forest",
    "model_version": "Random Forest v2.0",
    "best_cv_auc":   0.8266,
    "best_params":   {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt"},
    "training_samples": 10000,
    "models": {
        "Random Forest":  {"accuracy": 0.7383, "f1": 0.6741, "auc": 0.8142, "cv_auc": 0.8268},
        "Logistic Regr.": {"accuracy": 0.7457, "f1": 0.6872, "auc": 0.8243, "cv_auc": 0.8328},
        "K-NN (k=7)":     {"accuracy": 0.6990, "f1": 0.6146, "auc": 0.7421, "cv_auc": 0.7592},
        "Decision Tree":  {"accuracy": 0.6893, "f1": 0.6133, "auc": 0.6798, "cv_auc": 0.6663},
    },
}

try:
    import joblib

    # Load best model (Random Forest)
    best_model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))

    # Load encoders
    enc_path = os.path.join(MODEL_DIR, 'encoders.joblib')
    if os.path.exists(enc_path):
        encoders = joblib.load(enc_path)
    elif os.path.exists(os.path.join(MODEL_DIR, 'encoders.pkl')):
        encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))

    # Load feature columns
    fc_path = os.path.join(MODEL_DIR, 'feat_cols.pkl')
    if os.path.exists(fc_path):
        feat_cols = joblib.load(fc_path)

    # Load scaler
    sc_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    if os.path.exists(sc_path):
        scaler = joblib.load(sc_path)

    # Load real model stats if saved
    stats_path = os.path.join(MODEL_DIR, 'model_stats.pkl')
    if os.path.exists(stats_path):
        saved = joblib.load(stats_path)
        MODEL_STATS.update(saved)

    # Load model meta for real numbers
    meta_path = os.path.join(MODEL_DIR, 'model_meta.pkl')
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        MODEL_STATS['best_model']  = meta.get('best', 'Random Forest')
        MODEL_STATS['best_params'] = meta.get('best_params', {})
        MODEL_STATS['best_cv_auc'] = float(meta.get('best_cv_auc', 0.8266))
        # Convert numpy floats to python floats
        MODEL_STATS['models'] = {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in meta.get('results', MODEL_STATS['models']).items()
        }

    MODEL_STATS['ml_ready'] = True
    ML_READY = True
    rf = MODEL_STATS['models'].get('Random Forest', {})
    print(f"✅ ML models loaded — Best: {MODEL_STATS['best_model']}")
    print(f"   RF Accuracy : {rf.get('accuracy',0)*100:.2f}%")
    print(f"   RF AUC-ROC  : {rf.get('auc',0):.4f}")

except Exception as e:
    print(f"⚠️  ML models not loaded: {e}")
    print("   Using Rule-Based Engine as fallback.")

# ── HELPER ──
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ════════════════════════════════════════════════
#  PAGES
# ════════════════════════════════════════════════
@app.route('/')
def home():
    return render_template('index.html')

# ════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    first_name    = data.get('first_name','').strip()
    last_name     = data.get('last_name','').strip()
    email         = data.get('email','').strip().lower()
    username      = data.get('username','').strip()
    password      = data.get('password','')
    org           = data.get('organization','').strip()
    pref_airline  = data.get('preferred_airline', None)
    pref_dep      = data.get('preferred_dep_airport', None)
    pref_arr      = data.get('preferred_arr_airport', None)
    pref_aircraft = data.get('preferred_aircraft_type', None)

    if not all([first_name, last_name, email, username, password]):
        return jsonify({"success": False, "message": "All fields are required"}), 400
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO users
               (first_name,last_name,email,username,password,organization,
                preferred_airline,preferred_dep_airport,preferred_arr_airport,
                preferred_aircraft_type)
               VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (first_name,last_name,email,username,hash_password(password),org,
             pref_airline,pref_dep,pref_arr,pref_aircraft)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        session['user_id']    = user_id
        session['user_name']  = f"{first_name} {last_name}"
        session['user_email'] = email
        return jsonify({"success":True,"message":"Account created!",
                        "user":{"name":f"{first_name} {last_name}","email":email}})
    except Exception as e:
        print("SIGNUP ERROR:", e)
        return jsonify({"success":False,"message":"Email already exists"}), 409

@app.route('/api/signin', methods=['POST'])
def signin():
    data     = request.get_json()
    email    = data.get('email','').strip().lower()
    password = data.get('password','')
    if not all([email, password]):
        return jsonify({"success":False,"message":"Email and password required"}), 400
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email=? AND password=?',
                   (email, hash_password(password)))
    user = cursor.fetchone()
    conn.close()
    if user:
        session['user_id']    = user['id']
        session['user_name']  = f"{user['first_name']} {user['last_name']}"
        session['user_email'] = user['email']
        return jsonify({"success":True,"message":"Signed in!",
                        "user":{"name":f"{user['first_name']} {user['last_name']}",
                                "email":user['email']}})
    return jsonify({"success":False,"message":"Invalid email or password"}), 401

@app.route('/api/signout', methods=['POST'])
def signout():
    session.clear()
    return jsonify({"success":True,"message":"Signed out"})

@app.route('/api/user', methods=['GET'])
def get_user():
    if 'user_id' in session:
        return jsonify({"success":True,"logged_in":True,
                        "user":{"name":session.get('user_name'),
                                "email":session.get('user_email')}})
    return jsonify({"success":True,"logged_in":False})

# ════════════════════════════════════════════════
#  PASSWORD RECOVERY
# ════════════════════════════════════════════════
@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data  = request.get_json()
    email = data.get('email','').strip().lower()
    if not email:
        return jsonify({"success":False,"message":"Email is required"}), 400
    
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT email FROM users WHERE email=?', (email,))
    user = cursor.fetchone()
    conn.close()
    
    # We return success even if user not found to prevent user enumeration
    # but in this demo we'll be helpful.
    if user:
        return jsonify({"success":True,"message":"Recovery simulation started. Please verify your organization to reset password."})
    return jsonify({"success":False,"message":"Email not found in our records."}), 404

@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data     = request.get_json()
    email    = data.get('email','').strip().lower()
    org      = data.get('organization','').strip()
    new_pass = data.get('new_password','')
    
    if not all([email, org, new_pass]):
        return jsonify({"success":False,"message":"Email, Organization and New Password are required"}), 400
    
    if len(new_pass) < 8:
        return jsonify({"success":False,"message":"Password must be at least 8 characters"}), 400

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email=? AND organization=?', (email, org))
    user = cursor.fetchone()
    
    if not user:
        conn.close()
        return jsonify({"success":False,"message":"Verification failed. Email and Organization do not match."}), 401
    
    try:
        cursor.execute('UPDATE users SET password=? WHERE id=?', (hash_password(new_pass), user['id']))
        conn.commit()
        conn.close()
        return jsonify({"success":True,"message":"Password updated successfully! You can now sign in."})
    except Exception as e:
        conn.close()
        return jsonify({"success":False,"message":str(e)}), 500

# ════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    data       = request.get_json()
    weather    = data.get('weather_severity', 5)
    congestion = data.get('congestion', 5)
    wind       = data.get('wind_speed', 12)
    visibility = data.get('visibility', 10)
    hour       = data.get('dep_hour', 8)
    month      = data.get('month', 6)
    dow        = data.get('day_of_week', 2)
    airline    = data.get('airline', 'AA')
    distance   = data.get('distance', 1000)

    if ML_READY and best_model is not None:
        try:
            # Encode airline
            airline_enc = 0
            if encoders and isinstance(encoders, dict) and 'airline' in encoders:
                try:
                    airline_enc = int(encoders['airline'].transform([airline])[0])
                except:
                    airline_enc = 0
            else:
                airline_map = {'AA':0,'AS':1,'B6':2,'DL':3,'NK':4,'UA':5,'WN':6}
                airline_enc = airline_map.get(airline, 0)

            input_map = {
                'weather_severity': weather,
                'congestion':       congestion,
                'wind_speed':       wind,
                'visibility':       visibility,
                'dep_hour':         hour,
                'month':            month,
                'day_of_week':      dow,
                'airline_enc':      airline_enc,
                'distance':         distance,
            }

            if feat_cols is not None:
                row = [input_map.get(str(f), 0) for f in feat_cols]
            else:
                row = [weather, congestion, wind, visibility,
                       hour, month, dow, airline_enc, distance]

            X = np.array([row], dtype=float)
            if scaler is not None:
                X = scaler.transform(X)

            pred_class = int(best_model.predict(X)[0])
            proba      = best_model.predict_proba(X)[0].tolist()

            delay_map = {0: 0, 1: 15, 2: 45}
            label_map = {0: 'On Time', 1: 'Minor Delay', 2: 'Major Delay'}
            delay      = delay_map.get(pred_class, 0)
            confidence = round(float(max(proba)) * 100)
            probability= round(float(sum(proba[1:])), 2)

            # Feature importance
            fi       = best_model.feature_importances_
            fl       = list(feat_cols) if feat_cols is not None else \
                       ['weather_severity','congestion','wind_speed','visibility',
                        'dep_hour','month','day_of_week','airline_enc','distance']
            def gfi(name):
                try: return round(float(fi[fl.index(name)]) * 100)
                except: return 20

            result = {
                "delay_minutes":   delay,
                "confidence":      confidence,
                "probability":     probability,
                "model_used":      "Random Forest v2.0",
                "predicted_class": label_map[pred_class],
                "ml_powered":      True,
                "factors": {
                    "weather":     gfi('weather_severity'),
                    "congestion":  gfi('congestion'),
                    "carrier":     gfi('airline_enc'),
                    "time_of_day": gfi('dep_hour'),
                }
            }
        except Exception as e:
            print(f"ML predict error: {e}")
            result = _rule_based(weather, congestion, wind, visibility, hour, airline)
    else:
        result = _rule_based(weather, congestion, wind, visibility, hour, airline)

    # Save to DB
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO predictions
            (user_id,dep_airport,arr_airport,airline,aircraft_type,
             dep_hour,weather_severity,congestion,wind_speed,visibility,
             delay_minutes,confidence,probability,model_used)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (session.get('user_id'), data.get('dep_airport'),
             data.get('arr_airport'), airline, data.get('aircraft_type'),
             hour, weather, congestion, wind, visibility,
             result['delay_minutes'], result['confidence'],
             result['probability'], result['model_used']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB save warning: {e}")

    return jsonify(result)


def _rule_based(weather, congestion, wind, visibility, hour, airline):
    bias  = {'AA':5,'DL':-3,'UA':4,'WN':8,'B6':3,'AS':-5,'NK':12}
    delay = (weather*3.5)+(congestion*2.8)
    delay += (wind-20)*0.8 if wind>20 else 0
    delay += (5-visibility)*6 if visibility<5 else 0
    delay += 18 if 15<=hour<=19 else 0
    delay += bias.get(airline,0)
    delay  = max(0, round(delay*0.28))
    return {
        "delay_minutes":   delay,
        "confidence":      min(round(88+weather*0.5),99),
        "probability":     round(min(delay/120,1.0),2),
        "model_used":      "Rule-Based Engine v1",
        "predicted_class": "On Time" if delay==0 else ("Minor Delay" if delay<35 else "Major Delay"),
        "ml_powered":      False,
        "factors": {
            "weather":     round((weather/10)*100),
            "congestion":  round((congestion/10)*100),
            "carrier":     min(round(30+bias.get(airline,0)*2),99),
            "time_of_day": 60 if 15<=hour<=19 else 20,
        }
    }

# ════════════════════════════════════════════════
#  MODEL STATS — powers chart panel
# ════════════════════════════════════════════════
@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    m  = MODEL_STATS.get('models', {})
    rf = m.get('Random Forest',  {})
    lr = m.get('Logistic Regr.', {})
    kn = m.get('K-NN (k=7)',     {})
    dt = m.get('Decision Tree',  {})
    return jsonify({
        "ml_ready":      MODEL_STATS.get('ml_ready', False),
        "best_model":    MODEL_STATS.get('best_model', 'Random Forest'),
        "model_version": MODEL_STATS.get('model_version', 'Random Forest v2.0'),
        "best_cv_auc":   round(MODEL_STATS.get('best_cv_auc', 0.8266), 4),
        "best_params":   MODEL_STATS.get('best_params', {}),
        "training_samples": int(MODEL_STATS.get('training_samples', 10000)),
        "models": {
            "random_forest":       {"accuracy": round(rf.get('accuracy',0.7383)*100,2), "f1": round(rf.get('f1',0.6741),4), "auc": round(rf.get('auc',0.8142),4), "cv_auc": round(rf.get('cv_auc',0.8268),4)},
            "logistic_regression": {"accuracy": round(lr.get('accuracy',0.7457)*100,2), "f1": round(lr.get('f1',0.6872),4), "auc": round(lr.get('auc',0.8243),4), "cv_auc": round(lr.get('cv_auc',0.8328),4)},
            "knn":                 {"accuracy": round(kn.get('accuracy',0.6990)*100,2), "f1": round(kn.get('f1',0.6146),4), "auc": round(kn.get('auc',0.7421),4), "cv_auc": round(kn.get('cv_auc',0.7592),4)},
            "decision_tree":       {"accuracy": round(dt.get('accuracy',0.6893)*100,2), "f1": round(dt.get('f1',0.6133),4), "auc": round(dt.get('auc',0.6798),4), "cv_auc": round(dt.get('cv_auc',0.6663),4)},
        },
        "plots": {
            "confusion_matrix":   "/static/plots/confusion_matrix.png",
            "feature_importance": "/static/plots/feature_importance.png",
            "precision_recall":   "/static/plots/precision_recall.png",
            "rf_metrics":         "/static/plots/rf_metrics.png",
            "roc_curves":         "/static/plots/roc_curves.png",
        }
    })

# ════════════════════════════════════════════════
#  HISTORY
# ════════════════════════════════════════════════
@app.route('/api/history', methods=['GET'])
def get_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"success":False,"message":"Not logged in"}), 401
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10',
        (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return jsonify({"success":True,"history":[{
        "route":         f"{r['dep_airport']} → {r['arr_airport']}",
        "airline":       r['airline'],
        "delay_minutes": r['delay_minutes'],
        "confidence":    r['confidence'],
        "date":          r['created_at']
    } for r in rows]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)