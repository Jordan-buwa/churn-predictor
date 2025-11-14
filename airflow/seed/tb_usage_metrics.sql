CREATE TABLE usage_metrics (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    revenue FLOAT,
    mou FLOAT,
    recchrge FLOAT,
    directas FLOAT,
    overage FLOAT,
    roam FLOAT
);
