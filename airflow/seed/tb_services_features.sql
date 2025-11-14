CREATE TABLE services_features (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    mailord INT,
    mailres INT,
    mailflag INT,

    travel INT,
    pcown INT,
    webcap INT
);
