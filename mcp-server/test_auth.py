#!/usr/bin/env python3
"""
Simple test script to verify bearer token authentication.

Usage:
    # Start server in one terminal:
    python server.py --http

    # Run tests in another terminal:
    python test_auth.py
"""

import requests
import sys


def test_no_auth():
    """Test request without Authorization header."""
    print("Test 1: Request without auth header...")
    response = requests.get("http://localhost:8001/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    if response.status_code == 401:
        print("  ✓ Correctly rejected")
    else:
        print("  ✗ Should have been rejected")
    print()


def test_invalid_auth():
    """Test request with invalid token."""
    print("Test 2: Request with invalid token...")
    headers = {"Authorization": "Bearer wrong-key"}
    response = requests.get("http://localhost:8001/health", headers=headers)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    if response.status_code == 401:
        print("  ✓ Correctly rejected")
    else:
        print("  ✗ Should have been rejected")
    print()


def test_valid_auth():
    """Test request with valid token."""
    print("Test 3: Request with valid token...")
    headers = {"Authorization": "Bearer dev-key-jtbd"}
    response = requests.get("http://localhost:8001/health", headers=headers)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    if response.status_code == 200:
        print("  ✓ Correctly accepted")
    else:
        print("  ✗ Should have been accepted")
    print()


def test_malformed_header():
    """Test request with malformed auth header."""
    print("Test 4: Request with malformed header...")
    headers = {"Authorization": "NotBearer dev-key-jtbd"}
    response = requests.get("http://localhost:8001/health", headers=headers)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    if response.status_code == 401:
        print("  ✓ Correctly rejected")
    else:
        print("  ✗ Should have been rejected")
    print()


def test_ready_endpoint():
    """Test ready endpoint with valid auth."""
    print("Test 5: Ready endpoint with valid token...")
    headers = {"Authorization": "Bearer dev-key-jtbd"}
    response = requests.get("http://localhost:8001/ready", headers=headers)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    if response.status_code == 200:
        print("  ✓ Correctly accepted")
    else:
        print("  ✗ Should have been accepted")
    print()


if __name__ == "__main__":
    print("Bearer Token Authentication Tests")
    print("=" * 50)
    print()

    try:
        # Run all tests
        test_no_auth()
        test_invalid_auth()
        test_valid_auth()
        test_malformed_header()
        test_ready_endpoint()

        print("=" * 50)
        print("All tests completed!")
        print()
        print("Note: Make sure the server is running with:")
        print("  python server.py --http")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Make sure the server is running with:")
        print("  python server.py --http")
        sys.exit(1)
