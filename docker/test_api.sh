#!/bin/bash

curl -XPOST -H "Content-Type:application/json" http://localhost:5000/api/score -d '{
    "urls" : [
        "http://fakesite.com/test.jpg"
    ]
}' | jq .
