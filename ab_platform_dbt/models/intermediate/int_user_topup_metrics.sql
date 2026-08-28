select
    user_id,
    sum(amount) as total_amount,
    count(*) as tx_count,
    sum(case when is_fraud then 1 else 0 end) as fraud_count,
    sum(case when is_chargeback then 1 else 0 end) as chargeback_count
from {{ ref('stg_topup_transactions') }}
group by user_id
