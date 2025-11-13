CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    truck INT,
    rv INT,
    mcycle INT
);
