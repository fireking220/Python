import json
import requests
import mysql.connector

cursor = None
mydb = None


def main():
    global cursor
    global mydb
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="wooloo4Life%")
    cursor = mydb.cursor(buffered=True)
    cursor.execute("CREATE DATABASE IF NOT EXISTS pokedex")
    cursor.execute("USE pokedex")
    create_type_table()
    create_move_table()
    create_pokemon_table()
    insert_type()
    insert_pokemon()
    cursor.close()


# creates the pokemon table, if it doesn't exist
def create_pokemon_table():
    statement = ("CREATE TABLE IF NOT EXISTS pokemon("
                 "ID INT NOT NULL,"
                 "Name VARCHAR(255) NOT NULL,"
                 "Height VARCHAR(255) NOT NULL,"
                 "Weight VARCHAR(255) NOT NULL,"
                 "Entry VARCHAR(255) NOT NULL,"
                 "Type1 INT NOT NULL,"
                 "Type2 INT,"
                 "PRIMARY KEY (ID, NAME),"
                 "FOREIGN KEY (Type1) REFERENCES type(ID),"
                 "FOREIGN KEY (Type2) REFERENCES type(ID))")
    cursor.execute(statement)


# creates the type table, if it doesn't exist
def create_type_table():
    statement = ("CREATE TABLE IF NOT EXISTS type("
                 "ID INT NOT NULL,"
                 "Name VARCHAR(255) NOT NULL,"
                 "PRIMARY KEY (ID))")
    cursor.execute(statement)


# creates the move table, if it doesn't exist
def create_move_table():
    statement = ("CREATE TABLE IF NOT EXISTS move("
                 "ID INT NOT NULL,"
                 "Name VARCHAR(255) NOT NULL,"
                 "MoveClass VARCHAR(255) NOT NULL,"
                 "Power INT,"
                 "Accuracy INT,"
                 "Type INT NOT NULL,"
                 "Description VARCHAR(255) NOT NULL,"
                 "PRIMARY KEY (ID),"
                 "CHECK (MoveClass = 'Status' OR MoveClass = 'Physical' OR MoveClass = 'Special'))")
    cursor.execute(statement)


def insert_pokemon():
    results = None
    pokemon = []
    pokemonId = None
    name = None
    height = None
    weight = None
    entry = None
    entry_en = None
    type1 = None
    type2 = None
    statement = "INSERT INTO pokemon (ID, Name, Height, Weight, Entry, Type1, Type2) VALUES(%s, %s, %s, %s, %s, %s, %s)"
    print(cursor)

    for i in range(1, 152):
        try:
            results = requests.get("https://pokeapi.co/api/v2/pokemon/" + str(i)).json()
            pokemonId = i
            name = (results['species'])['name']
            height = results['height']
            weight = results['weight']
            entry = requests.get((results['species'])['url']).json()
            # get the english pokedex entry
            for flavor_text in entry['flavor_text_entries']:
                if (flavor_text['language'])['name'] == 'en':
                    entry_en = flavor_text['flavor_text']
                    break
            type1 = ((results['types'][0])['type'])['name']
            # get rows for selected type1 and store in memory
            cursor.execute("SELECT ID FROM type WHERE Name='%s'" % (type1,))
            # fetch first element of first tuple from previous SQL execution
            type1 = cursor.fetchone()[0]
            if len(results['types']) > 1:
                type2 = ((results['types'][1])['type'])['name']
                cursor.execute("SELECT ID FROM type WHERE Name='%s'" % (type2,))
                # fetch first element of first tuple from previous SQL execution
                type2 = cursor.fetchone()[0]
                # swap typeID's because PokeAPI got them reversed for two types (Silly PokeAPI~)
                temp = type1
                type1 = type2
                type2 = temp
            pokemon.append((pokemonId, name, height, weight, entry_en, type1, type2))
            # reset type 2
            type2 = None
        except json.decoder.JSONDecodeError:
            print("ERROR FETCHING results!")
    try:
        cursor.executemany(statement, pokemon)
        mydb.commit()
        print("ADDED POKEMON")
    # else rollback
    except mysql.connector.errors.IntegrityError:
        print("ERROR ADDING POKEMON, POKEMON ALREADY EXIST (INTEGRITY ERROR), ROLLING BACK...")
        mydb.rollback()


# insert types into database
def insert_type():
    types = []
    results = requests.get("https://pokeapi.co/api/v2/type").json()
    statement = "INSERT INTO type (ID, Name) VALUES (%s, %s)"

    # build types list of tuples
    for i in range(0, 18):
        types.append((i, ((results['results'])[i])['name']))

    # attempt to insert into database
    try:
        cursor.executemany(statement, types)
        mydb.commit()
        print("ADDED TYPES")
    # else rollback
    except mysql.connector.errors.IntegrityError:
        print("ERROR ADDING TYPES, TYPES ALREADY EXIST (INTEGRITY ERROR), ROLLING BACK...")
        mydb.rollback()


if __name__ == "__main__":
    main()
