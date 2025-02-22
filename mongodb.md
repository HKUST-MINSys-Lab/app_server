# mongoDB
[doc](https://www.mongodb.com/zh-cn/docs/manual/tutorial/install-mongodb-on-ubuntu/)

After modified the config
```
sudo systemctl restart mongod
sudo systemctl cat mongod
sudo systemctl status mongod
```

# using `mongosh`
a db shell, where you can access the data.
```shell
mongosh
use app # switch to mobibox database
show collections
db["imu_wenyu li"].findOne() # find first item in the collection
```
[CRUD DOC](https://www.mongodb.com/zh-cn/docs/mongodb-shell/crud/)