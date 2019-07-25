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
sed -i pubtrends/config.properties -E "s/version = [^\n]*/version = ${FULL_VERSION}/g"
sed -i pubtrends/config.properties -E "s/url = [^\n]*/url = ${url}/g"
sed -i pubtrends/config.properties -E "s/port = [^\n]*/port = ${port}/g"
sed -i pubtrends/config.properties -E "s/username = [^\n]*/username = ${username}/g"
# TODO BAD idea to store password like this
sed -i pubtrends/config.properties -E "s/password = [^\n]*/password = ${password}/g"
sed -i pubtrends/config.properties -E "s/database = [^\n]*/database = ${database}/g"
cp docker-compose.yml pubtrends/
cp -r models pubtrends/

# Create distributive tar.gz
rm -r dist
mkdir -p dist
tar -zcvf dist/pubtrends-${VERSION_BUILD}.tar.gz pubtrends
