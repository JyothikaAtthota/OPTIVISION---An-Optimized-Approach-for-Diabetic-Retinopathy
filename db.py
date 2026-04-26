"""
Optivision — SQLite Database Layer
Tables: patients | scans | model_metrics
Run:  python database/db.py   to initialise & seed
"""

import sqlite3, os, json, uuid, random
from datetime import datetime, timedelta

DB_FILE = os.path.join(os.path.dirname(__file__), "optivision.db")

# ─────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id                TEXT    PRIMARY KEY,
    name              TEXT    NOT NULL,
    age               INTEGER,
    gender            TEXT    CHECK(gender IN ('Male','Female','Other')),
    diabetes_duration INTEGER,
    hba1c             REAL,
    blood_pressure    TEXT,
    email             TEXT,
    phone             TEXT,
    created_at        TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS scans (
    id                  TEXT    PRIMARY KEY,
    patient_id          TEXT    REFERENCES patients(id),
    image_filename      TEXT,
    predicted_class     INTEGER NOT NULL CHECK(predicted_class BETWEEN 0 AND 4),
    confidence          REAL    NOT NULL,
    risk_score          REAL    NOT NULL,
    class_probabilities TEXT    NOT NULL,
    recommendations     TEXT    NOT NULL,
    notes               TEXT,
    created_at          TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch       INTEGER NOT NULL,
    train_loss  REAL,
    val_loss    REAL,
    train_acc   REAL,
    val_acc     REAL,
    precision_s REAL,
    recall_s    REAL,
    f1_score    REAL,
    recorded_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_scans_class   ON scans(predicted_class);
"""

DR_CLASSES = {0:"No DR",1:"Mild DR",2:"Moderate DR",3:"Severe DR",4:"Proliferative DR"}

RECOMMENDATIONS = {
    0: ["Annual retinal examination","Maintain HbA1c < 7%","Regular blood glucose monitoring","Healthy diet & exercise"],
    1: ["Retinal exam every 9 months","Optimise blood glucose control","Blood pressure management","Consult ophthalmologist"],
    2: ["Retinal exam every 6 months","Immediate ophthalmologist referral","Strict glycaemic control","Consider anti-VEGF evaluation"],
    3: ["Urgent ophthalmologist referral within 1 week","Monthly monitoring","Pan-retinal photocoagulation evaluation","Intensive systemic control"],
    4: ["EMERGENCY vitreoretinal referral","Consider immediate laser/surgery","Intensive inpatient metabolic management","Same-day ophthalmologist contact"],
}

TRAINING_METRICS = [
    (1, 1.842,1.763,0.412,0.431,0.387,0.401,0.372),
    (2, 1.561,1.498,0.541,0.558,0.512,0.527,0.501),
    (3, 1.298,1.267,0.623,0.641,0.598,0.614,0.589),
    (4, 1.102,1.124,0.689,0.672,0.651,0.667,0.643),
    (5, 0.967,1.031,0.724,0.709,0.691,0.706,0.682),
    (6, 0.851,0.947,0.758,0.741,0.724,0.738,0.716),
    (7, 0.764,0.891,0.783,0.769,0.752,0.765,0.744),
    (8, 0.698,0.854,0.806,0.793,0.777,0.789,0.768),
    (9, 0.641,0.824,0.824,0.812,0.798,0.809,0.789),
    (10,0.597,0.801,0.839,0.828,0.814,0.824,0.806),
    (11,0.558,0.783,0.851,0.841,0.827,0.836,0.819),
    (12,0.524,0.771,0.861,0.851,0.838,0.847,0.831),
    (13,0.493,0.764,0.869,0.858,0.846,0.854,0.839),
    (14,0.467,0.759,0.876,0.864,0.853,0.861,0.847),
    (15,0.444,0.757,0.882,0.869,0.859,0.866,0.852),
]

SEED_PATIENTS = [
    ("Priya Sharma",52,"Female",8, 7.8,"130/82","priya@mail.com","9876543210"),
    ("Ravi Kumar",  61,"Male",  14,9.2,"148/94","ravi@mail.com", "9876543211"),
    ("Ananya Reddy",44,"Female",5, 6.9,"122/78","ananya@mail.com","9876543212"),
    ("Suresh Patel",68,"Male",  19,10.4,"152/97","suresh@mail.com","9876543213"),
    ("Kavya Nair",  37,"Female",3, 7.1,"118/74","kavya@mail.com", "9876543214"),
    ("Arjun Mehta", 55,"Male",  10,8.6,"138/88","arjun@mail.com", "9876543215"),
    ("Deepa Iyer",  49,"Female",7, 7.4,"126/80","deepa@mail.com", "9876543216"),
    ("Vikram Singh",63,"Male",  16,9.8,"144/92","vikram@mail.com","9876543217"),
    ("Sneha Joshi", 41,"Female",4, 7.0,"120/76","sneha@mail.com", "9876543218"),
    ("Harish Bhat", 58,"Male",  11,8.1,"134/86","harish@mail.com","9876543219"),
]


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def insert_patient(conn, d):
    pid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO patients(id,name,age,gender,diabetes_duration,hba1c,blood_pressure,email,phone) VALUES(?,?,?,?,?,?,?,?,?)",
        (pid,d["name"],d.get("age"),d.get("gender"),d.get("diabetes_duration"),d.get("hba1c"),d.get("blood_pressure"),d.get("email"),d.get("phone"))
    ); conn.commit(); return pid

def insert_scan(conn, d):
    sid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO scans(id,patient_id,image_filename,predicted_class,confidence,risk_score,class_probabilities,recommendations,notes) VALUES(?,?,?,?,?,?,?,?,?)",
        (sid,d.get("patient_id"),d.get("image_filename"),d["predicted_class"],d["confidence"],d["risk_score"],
         json.dumps(d["class_probabilities"]),json.dumps(d["recommendations"]),d.get("notes"))
    ); conn.commit(); return sid

def get_all_patients(conn):
    return conn.execute("SELECT p.*,COUNT(s.id) AS scan_count FROM patients p LEFT JOIN scans s ON p.id=s.patient_id GROUP BY p.id ORDER BY p.created_at DESC").fetchall()

def get_patient(conn, pid):
    return conn.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()

def get_patient_scans(conn, pid):
    return conn.execute("SELECT * FROM scans WHERE patient_id=? ORDER BY created_at DESC",(pid,)).fetchall()

def get_recent_scans(conn, limit=10):
    return conn.execute("SELECT s.*,p.name AS patient_name FROM scans s LEFT JOIN patients p ON s.patient_id=p.id ORDER BY s.created_at DESC LIMIT ?",(limit,)).fetchall()

def get_dashboard_stats(conn):
    ts = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    tp = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    ac = conn.execute("SELECT AVG(confidence) FROM scans").fetchone()[0] or 0
    dist = conn.execute("SELECT predicted_class,COUNT(*) FROM scans GROUP BY predicted_class").fetchall()
    metrics = conn.execute("SELECT * FROM model_metrics ORDER BY epoch").fetchall()
    return ts,tp,ac,dist,metrics


def init_and_seed():
    print(f"\n🗄  Optivision DB → {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    conn.executescript(SCHEMA)

    if conn.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]==0:
        conn.executemany("INSERT INTO model_metrics(epoch,train_loss,val_loss,train_acc,val_acc,precision_s,recall_s,f1_score) VALUES(?,?,?,?,?,?,?,?)",TRAINING_METRICS)
        print(f"  ✓ {len(TRAINING_METRICS)} metric rows")

    if conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]==0:
        w=[0.49,0.07,0.27,0.05,0.12]
        pids=[]
        for row in SEED_PATIENTS:
            pid=str(uuid.uuid4()); pids.append(pid)
            conn.execute("INSERT INTO patients(id,name,age,gender,diabetes_duration,hba1c,blood_pressure,email,phone) VALUES(?,?,?,?,?,?,?,?,?)",(pid,*row))
        for pid in pids:
            for _ in range(random.randint(1,4)):
                cls=random.choices(range(5),weights=w)[0]
                raw=[random.uniform(0.03,0.18) for _ in range(5)]; raw[cls]=random.uniform(0.58,0.93)
                total=sum(raw); probs=[round(v/total,4) for v in raw]
                conf=probs[cls]; risk=round(sum(i*p for i,p in enumerate(probs))/4*100,1)
                ts=(datetime.now()-timedelta(days=random.randint(0,400))).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("INSERT INTO scans(id,patient_id,image_filename,predicted_class,confidence,risk_score,class_probabilities,recommendations,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()),pid,f"fundus_{random.randint(10000,99999)}.png",cls,conf,risk,json.dumps(probs),json.dumps(RECOMMENDATIONS[cls]),ts))
        print(f"  ✓ {len(pids)} patients seeded")
    conn.commit(); conn.close()
    print("  ✅ DB ready\n")

if __name__=="__main__":
    init_and_seed()
    conn=sqlite3.connect(DB_FILE)
    for t in ["patients","scans","model_metrics"]:
        n=conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n} rows")
    conn.close()
