#!/usr/bin/env bash
# Script for building tar.gz archive for deployment
# author Oleg.Shpynov os@jetbrains.com

version=0.2
build=development

for ARGUMENT in "$@"; do
    KEY=$(echo "${ARGUMENT}" | cut -f1 -d=)
    VALUE=$(echo "${ARGUMENT}" | cut -f2 -d=)

    case "$KEY" in
            version)    version=${VALUE} ;;
            build)      build=${VALUE} ;;
            *)          echo "Unknown KEY ${KEY} = ${VALUE}"; exit 1;;
    esac
done


VERSION_BUILD="${version}.${build}"
FULL_VERSION="${VERSION_BUILD} built on $(date +%F)"
PTV=pubtrends-${VERSION_BUILD}

# Update config version
mkdir -p ${PTV}
cp config.properties ${PTV}/
sed -E "s/version = [^\n]*/version = ${FULL_VERSION}/g" -i ${PTV}/config.properties
cp docker-compose.yml ${PTV}/
cp -r models ${PTV}/

# Create distributive tar.gz
rm -r dist
mkdir -p dist
tar -zcvf "dist/${PTV}.tar.gz" ${PTV}
# Cleanup
rm -f ${PTV}

# Move jar to dist if exists
if [[ -f build/libs/pubtrends-dev.jar ]]; then
  mv build/libs/pubtrends-dev.jar "dist/${PTV}.jar"
fi