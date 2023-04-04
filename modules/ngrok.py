from pyngrok import ngrok, conf, exception

def connect(token, port, region):
    account = None
    if token is None:
        token = 'None'
    else:
        if ':' in token:
            # token = authtoken:username:password
            account = token.split(':')[1] + ':' + token.split(':')[-1]
            token = token.split(':')[0]

    config = conf.PyngrokConfig(
        auth_token=token, region=region
    )
    
    # Guard for existing tunnels
    existing = ngrok.get_tunnels(pyngrok_config=config)
    if existing:
        for established in existing:
            # Extra configuration in the case that the user is also using ngrok for other tunnels
            if established.config['addr'][-4:] == str(port):
                public_url = existing[0].public_url
                print(f'ngrok has already been connected to localhost:{port}! URL: {public_url}\n'
                    'You can use this link after the launch is complete.')
                return
    
    try:
        if account is None:
            public_url = ngrok.connect(port, pyngrok_config=config, bind_tls=True).public_url
        else:
            public_url = ngrok.connect(port, pyngrok_config=config, bind_tls=True, auth=account).public_url
    except exception.PyngrokNgrokError:
        print(f'Invalid ngrok authtoken, ngrok connection aborted.\n'
              f'Your token: {token}, get the right one on https://dashboard.ngrok.com/get-started/your-authtoken')
    else:
        print(f'ngrok connected to localhost:{port}! URL: {public_url}\n'
               'You can use this link after the launch is complete.')
