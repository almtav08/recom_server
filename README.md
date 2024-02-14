# How to run the python server

1. The first thing you need to do is install the dependencies by running the following command:
   ```pip install -r requirements.txt```
2. Once the requirements are installed you can run the server by running the following command:
   ```uvicorn server:app --reload```
3. Before running the server, you must update the database url in the `schema.prisma` file. You can use a local database or a cloud database.
4. Once the database configuration is completed you can create the database by running the following commands:
   ```prisma generate``` and ```prisma db push```
5. If you want to insert some data into the database you can run the `load_dev_data.py` file with python.
6. The API documentation will be available at `http://{your_ip}:{your_port}/docs`

---

If you have any questions you can contact me at: [alemarti@uji.es](mailto:alemarti@uji.es)

---

This experiment was developed by Alex Martínez-Martínez as part of his research in the Institute of New Imaging Technologies at the Universitat Jaume I.

---

Coauthors: Raul Montoliu and Inmaculada Remolar.

---