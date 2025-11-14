CREATE TABLE tenure_info (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    months INT,
    uniqsubs INT,
    actvsubs INT
);
