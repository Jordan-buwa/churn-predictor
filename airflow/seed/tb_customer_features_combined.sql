-- Create/Refresh combined features table dynamically
CREATE TABLE IF NOT EXISTS combined_features AS
SELECT
    c.customer_id,
    c.customer_name,
    c.timestamp AS customer_timestamp,

    cl.churn,

    cm.changem,
    cm.changer,
    cm.dropvce,
    cm.blckvce,
    cm.unansvce,
    cm.custcare,
    cm.threeway,
    cm.mourec,
    cm.outcalls,
    cm.incalls,
    cm.peakvce,
    cm.opeakvce,
    cm.dropblk,
    cm.callfwdv,
    cm.callwait,

    cp.credita,
    cp.creditaa,
    cp.creditcd,
    cp.incmiss,

    d.age1,
    d.age2,
    d.children,
    d.income,
    d.ownrent,
    d.marryun,
    d.marryyes,

    ei.refurb,
    ei.phones,
    ei.models,
    ei.eqpdays,

    hh.truck AS household_truck,
    hh.rv AS household_rv,
    hh.mcycle AS household_mcycle,

    op.occprof,
    op.occler,
    op.occcrft,
    op.occstud,
    op.occhmkr,
    op.ocret,
    op.occself,

    pr.prizmrur,
    pr.prizmub,
    pr.prizmtwn,

    ri.refer,
    ri.retcall,
    ri.retcalls,
    ri.retaccpt,

    sf.mailord,
    sf.mailres,
    sf.mailflag,
    sf.travel,
    sf.pcown,
    sf.webcap,

    ti.months,
    ti.uniqsubs,
    ti.actvsubs,

    um.revenue,
    um.mou,
    um.recchrge,
    um.directas,
    um.overage,
    um.roam,

    v.truck AS vehicle_truck,
    v.rv AS vehicle_rv,
    v.mcycle AS vehicle_mcycle

FROM customer c
LEFT JOIN churn_label cl ON cl.customer_id = c.customer_id
LEFT JOIN call_metrics cm ON cm.customer_id = c.customer_id
LEFT JOIN credit_profile cp ON cp.customer_id = c.customer_id
LEFT JOIN demographics d ON d.customer_id = c.customer_id
LEFT JOIN equipment_info ei ON ei.customer_id = c.customer_id
LEFT JOIN household hh ON hh.customer_id = c.customer_id
LEFT JOIN occupation_profile op ON op.customer_id = c.customer_id
LEFT JOIN prizm_segment pr ON pr.customer_id = c.customer_id
LEFT JOIN retention_info ri ON ri.customer_id = c.customer_id
LEFT JOIN services_features sf ON sf.customer_id = c.customer_id
LEFT JOIN tenure_info ti ON ti.customer_id = c.customer_id
LEFT JOIN usage_metrics um ON um.customer_id = c.customer_id
LEFT JOIN vehicles v ON v.customer_id = c.customer_id;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_combined_features_customer_id
ON combined_features (customer_id);
