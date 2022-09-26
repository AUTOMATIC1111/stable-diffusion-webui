client_max_body_size 100M;
proxy_connect_timeout 6000s;
proxy_send_timeout    6000s;
proxy_read_timeout    6000s;
send_timeout          6000s;