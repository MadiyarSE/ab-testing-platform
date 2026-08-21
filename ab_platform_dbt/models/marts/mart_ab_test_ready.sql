select
    u.user_id,
    u.pilot,
    u.country,
    coalesce(m.metric, 0) as metric,
    coalesce(h.cov, 0) as cov
from {{ ref('stg_users') }} u
left join {{ ref('int_user_experiment_metrics') }} m on u.user_id = m.user_id
left join {{ ref('int_user_history_metrics') }} h on u.user_id = h.user_id
