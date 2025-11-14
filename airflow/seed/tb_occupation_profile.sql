CREATE TABLE occupation_profile (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    occprof INT,
    occcler INT,
    occcrft INT,
    occstud INT,
    occhmkr INT,
    ocret INT,
    occself INT
);
