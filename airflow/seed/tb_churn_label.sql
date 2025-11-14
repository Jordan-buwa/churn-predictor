CREATE TABLE churn_label (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),
    churn INT
);
