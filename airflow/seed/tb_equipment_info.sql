CREATE TABLE equipment_info (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    refurb INT,
    phones INT,
    models INT,
    eqpdays INT
);
