# Data Mining
This project was part of the "Data Mining" course at the University of Trento.

## Project description
A company is responsible for delivering different kinds of merchandise among different cities. In order to do such a thing, 
it hires drivers to whom assigns specific routes they should follow. For instance; go from Verona to Trento and deliver 
10 apples, then go to Venezia and deliver 4 bananas and 3 pens, etc. The problem resides in the fact that the driver often 
adds and omits cities and merchandise at his will, which causes a considerable discrepancy between what they are told to do 
(standard routes), and what they actually do (actual routes). Our job is to come up with a solution to the following three 
tasks, so that the company can use it to minimize the above discrepancy:
  1. Propose better standard routes
  2. For each driver, sort the existing standard routes depending on his tastes
  3. Propose an ideal standard route for every driver

## Instructions on how to run the program
The 'routemining.py' file contains the functions needed to find the new standard routes, sort the routes for a driver, and find the best route for a driver.
This file can be imported in python and then the functions can be used.

#### The file contains 4 main functions:
```findroutes(filename, limit_data=0, driver_id=0, prints=False)```
It finds new standard routes based on data in a JSON file. The output is a list of recommended routes.
filename is the directory of the JSON file with the actual routes in it.
limit_data may be used to limit the amount of routes taken out of the JSON file, these routes are randomly sampled without replacement, if 0 it will not be limmited
driver_id can be specified when the data should be minimized to only the data that one specific driver drove.
prints can be specified to see the progress of the function

```rankroutes(actual_routes_file, routes_to_sort_file, driver_id, limit_actual_routes=0, limit_routes_to_sort=0, prints=False, numperm3=128, findbest=0):```
It sorts the routes in the routes_to_sort_file based on the actual_routes_file. The output is a list of ids of sorted routes based on preference limited to 5
actual_routes_file is the directory of the JSON file with the actual routes in it.
routes_to_sort_file is the directory of the JSON file with the routes in it that need to be sorted.
limit_actual_routes may be used to limit the amount of routes taken out of the actual routes JSON file, these routes are randomly sampled without replacement, if 0 it will not be limmited
limit_routes_to_sort may be used to limit the amount of routes taken out of the routes to sort JSON file, these routes are randomly sampled without replacement, if 0 it will not be limmited
driver_id can be specified when the data should be minimized to only the data that one specific driver drove.
prints can be specified to see the progress of the function
numperm3 specify the number of permuations
findbest=0 is 0 by default but when changed to 1 the routes_to_sort_file is required to be a list of routes rather than the filename pointing to it

```findbestroute(actual_routes_file, driver_id, limit_data=0, prints=False,  printsalot=False, numperm=128):```
It gives the best route based on the data in the actual_routes_file The output is a single route that is considered as best
actual_routes_file is the directory of the JSON file with the actual routes in it.
limit_data may be used to limit the amount of routes taken out of the actual routes JSON file, these routes are randomly sampled without replacement, if 0 it will not be limmited
driver_id can be specified when the data should be minimized to only the data that one specific driver produced.
prints can be specified to see the progress of the function
printsalot can be specified to see more specifics of the function
numperm3 specify the number of permuations

```createfiles(filename_actual, filename_sort, destination_file, tasknumber ,limit_actual_routes=0, prints=False, printsalot=False):```
Creates the required file for a specific task number. The output is True
filename_actual is the directory of the JSON file with the actual routes in it.
filename_sort is the directory of the JSON file with the routes in it that need to be sorted. (only required for task 2)
destination_file is the location where the produced file should be stored

#### Variable explanations:
```tasknumber``` is the required task to be solved.<br>
- Task 1 is generating a file with new standard routes.<br>
- Task 2 is sorting routes that are in filename_sort based on filename_actual for every driver.<br>
- Task 3 is gnerating a file with the perfect route for every driver

```limit_actual_routes``` may be used to limit the amount of routes taken out of the actual routes JSON file, these routes are randomly sampled without replacement, if 0 it will not be limmited.<br>
```prints``` can be specified to see the progress of the function.<br>
```printsalot``` can be specified to see more specifics of the function.

## Team Members
- [Felix van der Waals](https://github.com/Bromsnortor)
- [Dennis Götz](https://github.com/dennismgoetz)
- [Unai Marin](https://github.com/unaimarin)
