# neuroglancer-python

## Installation

```bash
git clone https://github.com/MetaCell/neuroglancer-python
cd neuroglancer-python
pip install -e .

git clone -b develop https://github.com/MetaCell/neuroglancer/tree/develop
cd neuroglancer
npm i
npm run build-python
pip install .
```

## Usage

See [examples](examples) folder.

## Note

While PRs have not yet been merged, you can install the `neuroglancer` Python package from the forked repo of `neuroglancer` at https://github.com/MetaCell/neuroglancer/tree/develop. See the installation instructions above.

## Regression tests

```bash
cd tests
npm install
npm run test
```

You can update `constants.js` to point to some different data.

### Running individual tests

```bash
npm install jest --global
cd tests
jest MATCH --config=jest.config.js
```

where `MATCH` is a substring of the test name.