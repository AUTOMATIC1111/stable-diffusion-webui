import hashlib
import json
import os.path
import time

import filelock

from modules import shared
from modules.paths import data_path
from tools.mysql import get_mysql_cli, MySQLClient

cache_filename = os.path.join(data_path, "cache.json")
cache_data = None
is_worker = shared.cmd_opts.worker


def dump_cache(title, sha256):
    with filelock.FileLock(cache_filename+".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)
    # worker 模式不需要存
    if is_worker:
        return
    cli = get_mysql_cli()

    def write_model_hash(mysql_cli: MySQLClient):
        name, _ = os.path.splitext(os.path.basename(title))
        try:
            mysql_cli.execute_noquery_cmd('''
                CREATE TABLE IF NOT EXISTS `model` (
                  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
                  `filename` char(255) NOT NULL DEFAULT '' COMMENT '文件路径',
                  `alias` char(64) DEFAULT NULL COMMENT '别名',
                  `hash` char(64) DEFAULT NULL COMMENT '模型HASH',
                  `cover` char(255) DEFAULT NULL COMMENT '封面图',
                  `created_at` datetime DEFAULT NULL,
                  `updated_at` datetime DEFAULT NULL,
                  `deleted_at` datetime DEFAULT NULL,
                  `name` char(64) NOT NULL DEFAULT '' COMMENT '名称',
                  `user_id` int(11) NOT NULL DEFAULT '0' COMMENT '用户ID，0代表系统',
                  PRIMARY KEY (`id`),
                  UNIQUE KEY `name-user` (`name`,`user_id`),
                  UNIQUE KEY `hash` (`hash`)
                ) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;
            ''')
            r = mysql_cli.query("SELECT 1 FROM model WHERE name = %s", name)
            if r:
                mysql_cli.execute_noquery_cmd('UPDATE model SET hash=%s', sha256)
            else:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                mysql_cli.execute_noquery_cmd('''
                    INSERT INTO `model` (`filename`, `hash`, `created_at`, `updated_at`, `name`)
                    VALUES (%s, %s, %s, %s, %s);
                ''', (title, sha256, now, now, name))
        except Exception as ex:
            print(f"cannot write cache to mysql:{ex}")
        finally:
            mysql_cli.close()

    if cli:
        write_model_hash(cli)


def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename+".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s
    return s


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title):
    hashes = cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)

    if title not in hashes:
        def query_mysql():
            cli = get_mysql_cli()
            if cli:
                try:
                    if is_worker:
                        # OSS key 文件名就是hash
                        hash, _ = os.path.splitext(os.path.basename(title))
                        return hash
                    else:
                        name, _ = os.path.splitext(os.path.basename(title))
                        r = cli.query("SELECT hash FROM model WHERE name=%s LIMIT 1", name)
                    if r:
                        print(f"query cache from mysql:{r['hash']}")
                        return r.get('hash')
                except Exception as err:
                    print(f'query model hash err:{err}')
                finally:
                    cli.close()
            return None
        sha256 = query_mysql()
        if sha256:
            return sha256
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)

    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None

    return cached_sha256


def sha256(filename, title):
    hashes = cache("hashes")

    sha256_value = sha256_from_cache(filename, title)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    print(f"Calculating sha256 for {filename}: ", end='')
    sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    dump_cache(title, sha256_value)

    return sha256_value





