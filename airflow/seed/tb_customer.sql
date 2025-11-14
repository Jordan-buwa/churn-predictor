CREATE TABLE customer (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(255),
    traindata_flag VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW()
);
