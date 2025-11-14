CREATE TABLE prizm_segment (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    prizmrur INT,
    prizmub INT,
    prizmtwn INT
);
