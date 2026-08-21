select
    user_id,
    sum(revenue) as cov,
    count(*) as tx_count_hist
from {{ ref('stg_transactions_history') }}
group by user_id
