-- select product_id, product_name, list_price, model_year
-- from production.products where list_price  <= 900 and model_year = 2018

-- select *
-- from production.products where list_price  <= 900 and model_year = 2018

-- select * from sales.customers

-- select * from sales.customers where phone is null

-- select * from sales.customers where phone is not null

-- select * from sales.customers where state = 'TX' or state = 'NY'

-- select * from sales.customers where state in ('TX', 'NY', 'CA')

-- select * from sales.customers where customer_id between 8671 and 8675

-- select distinct state from sales.customers

-- select distinct city, state from sales.customers -- here distinct values maybe (ny-america, ny-africa) like distinct is distributed

-- select * from sales.customers where first_name like 'D%' -- any word starts with D

-- select * from sales.customers where first_name like '%D' -- any word ends with D

-- select * from sales.customers where first_name like '%D%' -- any word has letter with D

-- select * from sales.customers where first_name like '____'  -- any word of 4 letters

-- select * from sales.customers where first_name like '_d__'  -- any word of 4 letters and second letter is d


-- select * from sales.customers where first_name similar to '(A|e)%' -- any word start with A or E

-- select customer_id, first_name || ' ' || last_name as full_name , city from sales.customers

-- select first_name, last_name, email
-- from sales.customers
-- order by first_name asc;

-- select first_name, last_name, email
-- from sales.customers
-- order by first_name desc;


-- select product_id, product_name, list_price
-- from production.products
-- order by list_price desc;

-- select city, first_name, last_name 
-- from sales.customers 
-- order by city, first_name;

