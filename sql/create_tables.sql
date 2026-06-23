-- Credit Risk Feature Store Schema
-- German Credit Dataset (UCI / sklearn)

CREATE TABLE IF NOT EXISTS applicants (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    checking_status       TEXT,
    duration              INTEGER,
    credit_history        TEXT,
    purpose               TEXT,
    credit_amount         REAL,
    savings_status        TEXT,
    employment            TEXT,
    installment_commitment REAL,
    personal_status       TEXT,
    other_parties         TEXT,
    residence_since       REAL,
    property_magnitude    TEXT,
    age                   INTEGER,
    other_payment_plans   TEXT,
    housing               TEXT,
    existing_credits      REAL,
    job                   TEXT,
    num_dependents        REAL,
    own_telephone         TEXT,
    foreign_worker        TEXT,
    default_flag          INTEGER NOT NULL CHECK (default_flag IN (0, 1)),
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_applicants_default    ON applicants(default_flag);
CREATE INDEX IF NOT EXISTS idx_applicants_duration   ON applicants(duration);
CREATE INDEX IF NOT EXISTS idx_applicants_age        ON applicants(age);
CREATE INDEX IF NOT EXISTS idx_applicants_checking   ON applicants(checking_status);

-- Stores model predictions for monitoring
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id  INTEGER REFERENCES applicants(id),
    model_name    TEXT NOT NULL,
    score         INTEGER,
    probability   REAL,
    risk_tier     TEXT,
    predicted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
