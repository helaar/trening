#!/usr/bin/env python3
"""
Strava OAuth token generator script.

This script helps you generate a new Strava access token with the correct scopes
for the workout analyzer.

Requirements:
- Your Strava app's Client ID and Client Secret
- A web browser
"""

import os
import sys
import webbrowser
import urllib.parse
from pathlib import Path
import requests
from dotenv import load_dotenv

def main():
    print("Strava Access Token Generator")
    print("=" * 40)
    
    # Load existing .env if it exists
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    load_dotenv(dotenv_path=env_file)
    
    # Get Client ID and Secret from .env file
    client_id = os.getenv('STRAVA_CLIENT_ID')
    client_secret = os.getenv('STRAVA_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError("Please set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in your .env file.")
    
    print(f"Using Client ID from .env: {client_id}")
    
    # Step 1: Generate authorization URL
    redirect_uri = "http://localhost"  # You can use localhost for desktop apps
    scope = "activity:read_all"
    
    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&response_type=code"
        f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        f"&approval_prompt=force"
        f"&scope={scope}"
    )
    
    print(f"\nStep 1: Opening authorization URL in your browser...")
    print(f"URL: {auth_url}")
    
    try:
        webbrowser.open(auth_url)
        print("\nIf the browser didn't open, copy and paste the URL above.")
    except Exception:
        print("\nCould not open browser automatically. Please copy and paste this URL:")
        print(auth_url)
    
    print("\nAfter authorizing the application, you'll be redirected to a localhost URL.")
    print("The URL will look like: http://localhost/?state=&code=AUTHORIZATION_CODE&scope=read,activity:read_all")
    print("\nCopy the 'code' parameter from the URL.")
    
    # Step 2: Get authorization code from user
    auth_code = input("\nEnter the authorization code from the redirect URL: ").strip()
    if not auth_code:
        print("Authorization code is required!")
        return 1
    
    # Step 3: Exchange code for access token
    print("\nStep 2: Exchanging authorization code for access token...")
    
    token_url = "https://www.strava.com/oauth/token"
    token_data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': auth_code,
        'grant_type': 'authorization_code'
    }
    
    try:
        response = requests.post(token_url, data=token_data)
        response.raise_for_status()
        token_info = response.json()
        
        access_token = token_info.get('access_token')
        refresh_token = token_info.get('refresh_token')
        expires_at = token_info.get('expires_at')
        scope = token_info.get('scope')
        
        print("\n✅ Successfully obtained access token!")
        print(f"Access Token: {access_token}")
        print(f"Refresh Token: {refresh_token}")
        print(f"Scope: {scope}")
        print(f"Expires At: {expires_at}")
        
        # Step 4: Update .env file
        update_env = input("\nUpdate your .env file automatically? (y/n): ").lower().strip()
        if update_env in ['y', 'yes']:
            update_env_file(env_file, access_token, refresh_token, client_id, client_secret)
        else:
            print("\nManually add this to your .env file:")
            print(f"STRAVA_ACCESS_TOKEN={access_token}")
            print(f"STRAVA_REFRESH_TOKEN={refresh_token}")
            print(f"STRAVA_CLIENT_ID={client_id}")
            print(f"STRAVA_CLIENT_SECRET={client_secret}")
        
        return 0
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print("This usually means:")
        print("- Invalid authorization code (codes expire quickly)")
        print("- Wrong Client ID or Client Secret")
        print("- The code was already used")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

def update_env_file(env_path: Path, access_token: str, refresh_token: str, client_id: str, client_secret: str):
    """Update or create .env file with new tokens."""
    try:
        # Read existing .env content
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add token lines
        updated_lines = []
        tokens_added = {
            'STRAVA_ACCESS_TOKEN': False,
            'STRAVA_REFRESH_TOKEN': False,
            'STRAVA_CLIENT_ID': False,
            'STRAVA_CLIENT_SECRET': False
        }
        
        for line in env_lines:
            if line.startswith('STRAVA_ACCESS_TOKEN='):
                updated_lines.append(f'STRAVA_ACCESS_TOKEN={access_token}\n')
                tokens_added['STRAVA_ACCESS_TOKEN'] = True
            elif line.startswith('STRAVA_REFRESH_TOKEN='):
                updated_lines.append(f'STRAVA_REFRESH_TOKEN={refresh_token}\n')
                tokens_added['STRAVA_REFRESH_TOKEN'] = True
            elif line.startswith('STRAVA_CLIENT_ID='):
                updated_lines.append(f'STRAVA_CLIENT_ID={client_id}\n')
                tokens_added['STRAVA_CLIENT_ID'] = True
            elif line.startswith('STRAVA_CLIENT_SECRET='):
                updated_lines.append(f'STRAVA_CLIENT_SECRET={client_secret}\n')
                tokens_added['STRAVA_CLIENT_SECRET'] = True
            else:
                updated_lines.append(line)
        
        # Add missing tokens
        if not tokens_added['STRAVA_ACCESS_TOKEN']:
            updated_lines.append(f'STRAVA_ACCESS_TOKEN={access_token}\n')
        if not tokens_added['STRAVA_REFRESH_TOKEN']:
            updated_lines.append(f'STRAVA_REFRESH_TOKEN={refresh_token}\n')
        if not tokens_added['STRAVA_CLIENT_ID']:
            updated_lines.append(f'STRAVA_CLIENT_ID={client_id}\n')
        if not tokens_added['STRAVA_CLIENT_SECRET']:
            updated_lines.append(f'STRAVA_CLIENT_SECRET={client_secret}\n')
        
        # Write updated .env file
        with open(env_path, 'w') as f:
            f.writelines(updated_lines)
        
        print(f"✅ Updated {env_path}")
        
    except Exception as e:
        print(f"❌ Could not update .env file: {e}")
        print("Please manually update your .env file with the tokens shown above.")

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        sys.exit(1)