select
    u.user_id,
    u.pilot,
    u.country,
    coalesce(m.total_amount, 0) as total_amount,
    coalesce(m.tx_count, 0) as tx_count,
    coalesce(m.fraud_count, 0) as fraud_count,
    coalesce(m.chargeback_count, 0) as chargeback_count
from {{ ref('stg_users_topup') }} u
left join {{ ref('int_user_topup_metrics') }} m on u.user_id = m.user_id
