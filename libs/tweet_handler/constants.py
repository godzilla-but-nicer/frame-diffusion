from datetime import datetime


# we have to refer to these a bunch so lets put them in one place
immigration_keywords = "immigration|immigrants?|illegals|undocumented|illegal aliens?|migrants?|migration"

# these are the dates of the public tweets data
start_date = datetime.strptime("01-01-2018", "%m-%d-%Y")
end_date = datetime.strptime("12-31-2018", "%m-%d-%Y")