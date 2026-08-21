select
    user_id,
    sum(revenue) as metric,
    count(*) as tx_count
from {{ ref('stg_transactions_experiment') }}
group by user_id
