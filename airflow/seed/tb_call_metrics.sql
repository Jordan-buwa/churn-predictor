CREATE TABLE call_metrics (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customer(customer_id),

    changem FLOAT,
    changer FLOAT,
    dropvce FLOAT,
    blckvce FLOAT,
    unansvce FLOAT,
    custcare FLOAT,
    threeway FLOAT,

    mourec FLOAT,
    outcalls FLOAT,
    incalls FLOAT,

    peakvce FLOAT,
    opeakvce FLOAT,
    dropblk FLOAT,
    callfwdv FLOAT,
    callwait FLOAT
);
