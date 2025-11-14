CREATE TABLE demographics (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    age1 INT,
    age2 INT,
    children INT,
    income FLOAT,
    ownrent VARCHAR(20),

    marryun INT,
    marryyes INT
);
