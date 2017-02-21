#!/bin/bash

echo "trying image w/ flags, people and guns"
curl -XPOST -H "Content-Type:application/json" http://localhost:5000/api/score -d '{
    "urls" : [
        "http://i.telegraph.co.uk/multimedia/archive/03166/isil_3166282b.jpg",
        "http://i.telegraph.co.uk/multimedia/archive/03166/isil_3166282b.jpg"
    ]
}' | jq .

curl -XPOST -H "Content-Type:application/json" http://localhost:5000/api/score -d '{
    "urls" : [
        "http://i.fakesite.co.uk/multimedia/archive/03166/isil_3166282b.jpg"
    ]
}' | jq .
