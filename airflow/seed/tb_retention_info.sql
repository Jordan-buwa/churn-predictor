CREATE TABLE retention_info (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    refer INT,
    retcall INT,
    retcalls INT,
    retaccpt INT
);
