DROP TABLE IF EXISTS farmers;
DROP TABLE IF EXISTS consumers;

CREATE TABLE farmers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    place TEXT NOT NULL,
    mobile_number TEXT,
    email_id TEXT
);

CREATE TABLE consumers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    place TEXT NOT NULL,
    email_id TEXT
);
