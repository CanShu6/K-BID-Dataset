# K-BID-Dataset
These code and dataset files are used to reproduce K-BID, which is a recommendation model for broad information diffusion.

## Environment Requirement
The code has been tested running under Python 3.10.13. The required packages are as follows:
* torch == 2.1.1
* numpy == 1.26.2
* pandas == 2.1.3
* scipy == 1.11.4
* tqdm == 4.66.1


## Code 

The K-BID model can be found in https://github.com/CanShu6/K-BID.

## Datast
The data has already been processed.
We perform experiments on real-life e-commerce datasets sourced from the public Aliyun Tianchi Competition.
**The original dataset can be found in original-dataset.zip.**

Commodity sharing is the main form of social interaction for users in e-commerce platforms. The behavior of commodity sharing contains the information of users 'social interaction preferences based on commodities. This task focuses on the problem of dynamic link prediction in social graph. The quadruple (u, i, v, t) is used to represent the user's interactive behavior of item sharing at time t (time), where i (item) identifies a specific item and u (user) represents the invited user who initiated the item link sharing. v (voter) is the returning user who received and clicked on the item link. Therefore, link prediction in this task refers to predicting the corresponding backflow user v given the invited user u, item i and time t.


### Dataset Files
- `item_info.json`
  - Introduction: Item information data, each row is a json string, the specific field information is as follows.
  - Format: 
    - \<#item_id\>. Item id.
    - \<#cate_id\>. Item leaf category id.
    - \<#cate_level1_id\>. Item top-level category id.
    - \<#brand_id\>. Item brand id.
    - \<#shop_id\>. Item shop id.
  - MD5: ed0530a00b4629cec7a97f4e5562ecdc
- `user_info.json`
  - Introduction: User information data, each row is a json string, the specific field information is as follows.
  - Format: 
    - \<#user_id\>. User id, which contains all inviter_ids and voter_ids in the dataset.
    - \<#user_gender\>. User gender, 0 for female user, 1 for male user, unknown as -1.
    - \<#user_age\>. User age, which ranges from 1 to 8, with higher values indicating greater age and -1 for unknown.
    - \<#user_level\>. User's platform credit level, where the value ranges from 1 to 10, with a higher value indicating a higher platform credit level for the user.
  - MD5: 8537da64fe50f0c068f461f9c500d825	
- `item_share_preliminary_train/test_info.json`
  - Introduction: User dynamic item sharing data, each row is a json string, the specific field information is as follows.
  - Format: the same as `valid/invalid.txt`.
    - \<#item_id\>. Item id.
    - \<#cate_id\>. Item leaf category id.
    - \<#cate_level1_id\>. Item top-level category id.
    - \<#brand_id\>. Item brand id.
    - \<#shop_id\>. Item shop id.
  - MD5:
    - 29da9d5080c4ed258eac7ccb9151e863	preliminary_train.txt
    - 7b67e90541a378d7bfbc0db43e3f1b7d	test.txt
- `item_share_final_train/test_info.json`
  - Introduction/Format: the same as `item_share_preliminary_train/test_info.json`.
    - MD5: 
    - b042c0355c7fb2c181f11198b2a2097d	final_train.txt
    - 474ce795828a4cf5d9ee93506413e319	final_test.txt


### Acknowledgement
