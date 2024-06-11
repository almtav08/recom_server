# Overview of this project
This is an hybrid recommender system joining together collaborative filtering and knowledge-based filtering. It is also a result of the Horizon Europe Project e-DIPLOMA (Project number: 101061424, Call: HORIZON-CL2-2021-TRANSFORMATIONS-01, Duration: 01/09/2022 – 31/08/2025). This recommender is meant to work within LMS (it has been developed using Moodle as the selected LMS). For this reason you also need to check this moodle blocks for this recommender to work properly. Recommender Block: [https://github.com/almtav08/moodle_recommender_block](https://github.com/almtav08/moodle_recommender_block), and Logevent Plugin: [https://github.com/almtav08/moodle_recommender_plugin](https://github.com/almtav08/moodle_recommender_plugin). The block and the plugin need to be configured with the IP address of the recommender, so please change this in the respective places.

# How to run the python server

This python server is prepared for runing using conda and Ubuntu. If you want to work with other package manager such as pip or poetry or another OS take into account that specific packages (e.g., PyTorch) may cause issues.

1. The first thing you need to do is to prepare the python environment. Using conda this can be done by running the following command:
```conda env create -f environment.yml```.

2. Once the python environment is created, the next step should be to establish all environment variables. You can do this by creating a file called ```.env``` in the root of the project. This file should contain the following variables:
- ```PLATFORM_URL```: The URL of the moodle platform.
- ```PLATFORM_API_KEY```: The token of the moodle platform.
- ```DATABASE_URL```: The URL of the database.
- ```USER_API_PASS```: The password of the server API for the users.
- ```SEND_MESSAGES_FUNC```: The URL of the function that sends messages to the users.
- ```GET_USER_QUIZ_ATTEMPTS```: The URL of the function that gets the quiz attempts of the users.
- ```GET_USER_ATTEMPT_REVIEW```: The URL of the function that gets the review of the quiz attempts of the users.
- ```GET_COURSE_CONTENTS```: The URL of the function that gets the contents of the courses.
- ```GET_COURSE_INFO```: The URL of the function that gets the information of the courses.
- ```GET_ENROLLED_STUDENTS```: The URL of the function that gets the enrolled students of the courses.
- ```GET_COURSE_LOGS```: The URL of the function that gets the logs of the courses (this is the one establish from the recommender plugin).

3. Once this settings are ready the server can be run by using the following command:
```uvicorn main:app --host {your_ip} --port {your_port}```.

Please note that you should declare an IP address and a port for the server to execute. Also you can add the tag ```--reload``` if you want it to reexecute if you make any changes while it is running. Before running the server you first need to load the database and how the Knowledge Graph entities are related to each other. There are different ```load_{element}.py``` files that are meant to prepare this information. You should execute them in the following order:

1. ```load_resources_data.py```
2. ```load_users_data.py```
3. ```load_interaction_data.py```
4. ```load_db.py```
5. ```load_prevgraph_data.py```
6. ```load_repeatgraph_data.py```

The ```load_db.py``` must always be executed before the graphs loaders. While loading the resources data, a window will pop up and you will have to mark which resources of the moodle course should be part of the recommendation process. While loading the graphs you will have to indicate the relation items separated by comma in the corresponding textbox (e.g., Moodle ID: 1, Module Name: 'Introduction', Related Entities: 2,3,4).

---

If you have any questions you can contact me at: [alemarti@uji.es](mailto:alemarti@uji.es)

---

This system was developed by Alex Martínez-Martínez as part of his research in the Institute of New Imaging Technologies at the Universitat Jaume I.

---

Coauthors: Raul Montoliu and Inmaculada Remolar.

---