const URL = 'https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui-extensions/master/index.json';
const console = require('console');

const headers = {
  'content-type': 'application/json',
  'accept': 'application/vnd.github.v3+json',
  'user-agent': 'nodejs/fetch',
};

let logger = new console.Console({
  stdout: process.stdout,
  stderr: process.stderr,
  ignoreErrors: true,
  inspectOptions: {
    showHidden: false,
    depth: 5,
    colors: true,
    showProxy: true,
    maxArrayLength: 1024,
    maxStringLength: 10240,
    breakLength: process.stdout.columns,
    compact: 64,
    sorted: false,
    getters: false,
  }
});

const log = (...args) => logger.log(...args);

const http = async (url) => {
  try {
    const res = await fetch(url, { method: 'GET', headers })
    if (res.status !== 200) {
      const limit = parseInt(res.headers.get('x-ratelimit-remaining') || '0');
      if (limit === 0) log(`ratelimit: ${url}`, new Date(1000 * parseInt(res.headers.get('x-ratelimit-reset'))));
      else log(`error: ${res.status} ${res.statusText} ${url}`)
      return {};
    }
    const text = await res.text();
    const json = JSON.parse(text);
    return json;
  } catch (e) {
    log(`exception: ${e} ${url}`)
    return {};
  }
};

const head = async (url) => {
  let json = {};
  const res = await fetch(url, { method: 'GET', headers })
  if (res.status !== 200) return {};
  for (const h of res.headers) json[h[0]] = h[1];
  return json;
};

async function getDetails(extension) {
  return new Promise(async (resolve) => {
    let name = extension.url.replace('https://github.com/', '');
    if (name.endsWith('.git')) name = name.replace('.git', '');
    const ext = {
      name: extension.name,
      url: extension.url,
      description: extension.description.substring(0, 64),
      tags: extension.tags,
      added: new Date(extension.added),
    }
    if (ext.tags.includes('localization')) resolve(ext); // don't fetch details for localization tags
    else {
      const r = await http(`https://api.github.com/repos/${name}`);
      if (!r.full_name) resolve(ext);
      else {
        ext.created = new Date(r.created_at);
        ext.updated = new Date(r.pushed_at);
        ext.name = r.full_name;
        ext.description = (r.description || extension.description).substring(0, 96);
        ext.size = r.size;
        ext.stars = r.stargazers_count;
        ext.issues = r.open_issues_count;
        const h = await head(`https://api.github.com/repos/${r.full_name}/commits?per_page=1`); // get headers
        const commits = parseInt(h.link?.match(/\d+/g).pop() || '0'); // extract total commit count from pagination data
        ext.commits = commits;
        // const c = await http(`https://api.github.com/repos/${r.full_name}/traffic/clones?per=week`); // not allowed for non-owners
        // const clones = c.clones?.map((c) => c.count).reduce((avg, value, _, { length }) => avg + value / length, 0) || 0;
        // ext.clones: Math.round(clones),
        resolve(ext);
      }
    }
  })
}

async function main() {
  // const repos = await githubRepositories();
  if (process.env.GITHUB_TOKEN) {
    log('using github token');
    headers['authorization'] = `token ${process.env.GITHUB_TOKEN}`;
  } else {
    log('no github token set so low rate limiting will apply');
  }
  log(`fetching extensions index: ${URL}`)
  const index = await http(URL);
  const extensions = index?.extensions || [];
  log(`analyzing extensions: ${extensions.length}`)

  const promises = [];
  for (const e of extensions) {
    ext = getDetails(e);
    promises.push(ext);
  }
  let details = await Promise.all(promises);

  const word = process.argv[2]
  if (word) {
    if (details[0][word]) {
      log('sorting by field:', word);
      details.sort((a, b) => b[word] - a[word]);
    } else if (Object.keys(index?.tags || {}).includes(word)) {
      log('filtering by tag:', word);
      details = details.filter((a) => a.tags.includes(word));
    } else {
      log('filtering by keyword:', word);
      details = details.filter((a) => (a.name.includes(word) || a.description.includes(word)));
    }
  }
  details.length = Math.min(parseInt(process.argv[3] || 100000), details.length);
  log('extensions:');
  for (const ext of details) log(ext);
}

main();
