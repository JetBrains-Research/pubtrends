#!/usr/bin/env bash
# Script for building tar.gz archive for deployment
# author Oleg.Shpynov os@jetbrains.com

version=0.1
build=development
url=localhost
port=5432
username=biolabs
password=password
database=pubtrends

for ARGUMENT in "$@"; do

    KEY=$(echo ${ARGUMENT} | cut -f1 -d=)
    VALUE=$(echo ${ARGUMENT} | cut -f2 -d=)

    case "$KEY" in
            version)    version=${VALUE} ;;
            build)      build=${VALUE} ;;
            url)        url=${VALUE} ;;
            port)       port=${VALUE} ;;
            username)   username=${VALUE} ;;
            password)   password=${VALUE} ;;
            database)   database=${VALUE} ;;
            *)          echo "Unknown KEY ${KEY} = ${VALUE}"; exit 1;;
    esac
done


VERSION_BUILD="${version}.${build}"
FULL_VERSION="${VERSION_BUILD} built on $(date)"

# Update config
mkdir -p pubtrends
cp config.properties pubtrends/
sed -E "s/version = [^\n]*/version = ${FULL_VERSION}/g"     -i pubtrends/config.properties
sed -E "s/url = [^\n]*/url = ${url}/g"                      -i pubtrends/config.properties
sed -E "s/port = [^\n]*/port = ${port}/g"                   -i pubtrends/config.properties
sed -E "s/username = [^\n]*/username = ${username}/g"       -i pubtrends/config.properties
# TODO BAD idea to store password like this
sed -E "s/password = [^\n]*/password = ${password}/g"       -i pubtrends/config.properties
sed -E "s/database = [^\n]*/database = ${database}/g"       -i pubtrends/config.properties
cp docker-compose.yml pubtrends/
cp -r models pubtrends/

# Create distributive tar.gz
rm -r dist
mkdir -p dist
tar -zcvf dist/pubtrends-${VERSION_BUILD}.tar.gz pubtrends
