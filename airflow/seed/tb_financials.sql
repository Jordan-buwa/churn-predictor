CREATE TABLE financials (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    setprc FLOAT,
    setprcm FLOAT
);
