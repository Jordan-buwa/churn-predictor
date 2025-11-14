CREATE TABLE credit_profile (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    credita INT,
    creditaa INT,
    creditcd INT,
    incmiss INT
);
