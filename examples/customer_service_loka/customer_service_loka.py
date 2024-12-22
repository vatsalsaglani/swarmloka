import json
import sqlite3
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Union
from swarmloka import Loka, LocalLLM, Agent, ContextVariable

model_name = "gpt-4o"
llm = LocalLLM("https://api.openai.com/v1", "YOUR_API_KEY_HERE")


def create_tables_with_sample_data():
    conn = sqlite3.connect('customer_service.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,  -- Unique identifier for each customer
        first_name TEXT,                  -- Customer's first name
        last_name TEXT,                   -- Customer's last name
        email TEXT UNIQUE,                -- Customer's email address
        phone TEXT,                       -- Customer's phone number
        address TEXT,                     -- Customer's physical address
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Account creation date
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,     -- Unique identifier for each order
        customer_id INTEGER,              -- Reference to customer who placed the order
        order_date TIMESTAMP,             -- Date when order was placed
        total_amount DECIMAL(10,2),       -- Total order amount
        status TEXT,                      -- Order status (pending, completed, cancelled)
        shipping_address TEXT,            -- Delivery address
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        item_id INTEGER PRIMARY KEY,      -- Unique identifier for each order item
        order_id INTEGER,                 -- Reference to parent order
        product_name TEXT,                -- Name of the product
        quantity INTEGER,                 -- Quantity ordered
        unit_price DECIMAL(10,2),         -- Price per unit
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_interactions (
        interaction_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        interaction_type TEXT,        -- complaint, inquiry, feedback
        description TEXT,
        created_at TIMESTAMP,
        status TEXT,                  -- open, resolved, pending
        resolution_notes TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_status_history (
        history_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        status TEXT,                  -- processing, shipped, delivered
        status_date TIMESTAMP,
        notes TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_issues (
        issue_id INTEGER PRIMARY KEY,
        order_item_id INTEGER,        -- Reference to specific item in order
        customer_id INTEGER,
        issue_type TEXT,             -- defect, warranty, usage_help
        description TEXT,
        reported_date TIMESTAMP,
        resolution TEXT,
        FOREIGN KEY (order_item_id) REFERENCES order_items(item_id),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS payments (
        payment_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        amount DECIMAL(10,2),
        payment_method TEXT,          -- credit_card, paypal, bank_transfer
        payment_status TEXT,          -- pending, completed, failed
        transaction_date TIMESTAMP,
        refund_status TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS shipments (
        shipment_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        tracking_number TEXT,
        carrier TEXT,                 -- UPS, FedEx, USPS
        shipping_method TEXT,         -- standard, express, overnight
        estimated_delivery DATE,
        actual_delivery DATE,
        shipping_status TEXT,         -- in_transit, delivered, delayed
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS returns (
        return_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        customer_id INTEGER,
        return_reason TEXT,
        return_status TEXT,           -- requested, approved, completed
        created_at TIMESTAMP,
        refund_amount DECIMAL(10,2),
        return_tracking TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')

    sample_customers = [('John', 'Doe', 'john.doe@email.com', '555-0101',
                         '123 Main St', '2023-01-01'),
                        ('Jane', 'Smith', 'jane.smith@email.com', '555-0102',
                         '456 Oak Ave', '2023-01-02'),
                        ('Michael', 'Johnson', 'michael.j@email.com',
                         '555-0103', '789 Pine Rd', '2023-01-03'),
                        ('Emily', 'Brown', 'emily.b@email.com', '555-0104',
                         '321 Elm St', '2023-01-04'),
                        ('William', 'Davis', 'william.d@email.com', '555-0105',
                         '654 Maple Dr', '2023-01-05'),
                        ('Sarah', 'Wilson', 'sarah.w@email.com', '555-0106',
                         '987 Cedar Ln', '2023-01-06'),
                        ('David', 'Taylor', 'david.t@email.com', '555-0107',
                         '147 Birch Ave', '2023-01-07'),
                        ('Lisa', 'Anderson', 'lisa.a@email.com', '555-0108',
                         '258 Spruce St', '2023-01-08'),
                        ('Robert', 'Thomas', 'robert.t@email.com', '555-0109',
                         '369 Willow Rd', '2023-01-09'),
                        ('Jennifer', 'Moore', 'jennifer.m@email.com',
                         '555-0110', '741 Ash Dr', '2023-01-10'),
                        ('James', 'Martin', 'james.m@email.com', '555-0111',
                         '852 Oak St', '2023-01-11'),
                        ('Patricia', 'Jackson', 'patricia.j@email.com',
                         '555-0112', '963 Pine Ave', '2023-01-12'),
                        ('Richard', 'White', 'richard.w@email.com', '555-0113',
                         '159 Maple Ln', '2023-01-13'),
                        ('Linda', 'Harris', 'linda.h@email.com', '555-0114',
                         '357 Cedar St', '2023-01-14'),
                        ('Charles', 'Clark', 'charles.c@email.com', '555-0115',
                         '486 Elm Dr', '2023-01-15'),
                        ('Margaret', 'Lewis', 'margaret.l@email.com',
                         '555-0116', '729 Birch Rd', '2023-01-16'),
                        ('Joseph', 'Lee', 'joseph.l@email.com', '555-0117',
                         '813 Spruce Ave', '2023-01-17'),
                        ('Sandra', 'Walker', 'sandra.w@email.com', '555-0118',
                         '924 Willow St', '2023-01-18'),
                        ('Thomas', 'Hall', 'thomas.h@email.com', '555-0119',
                         '537 Pine Dr', '2023-01-19'),
                        ('Nancy', 'Young', 'nancy.y@email.com', '555-0120',
                         '648 Oak Ln', '2023-01-20')]

    cursor.executemany(
        '''
    INSERT OR IGNORE INTO customers (first_name, last_name, email, phone, address, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_customers)

    sample_orders = [(1, '2023-01-15', 150.00, 'completed', '123 Main St'),
                     (1, '2023-02-01', 75.50, 'pending', '123 Main St'),
                     (2, '2023-02-15', 299.99, 'completed', '456 Oak Ave'),
                     (3, '2023-02-20', 89.99, 'completed', '789 Pine Rd'),
                     (4, '2023-03-01', 199.50, 'pending', '321 Elm St'),
                     (5, '2023-03-05', 449.99, 'completed', '654 Maple Dr'),
                     (6, '2023-03-10', 129.99, 'cancelled', '987 Cedar Ln'),
                     (7, '2023-03-15', 899.99, 'completed', '147 Birch Ave'),
                     (8, '2023-03-20', 159.99, 'pending', '258 Spruce St'),
                     (9, '2023-03-25', 279.99, 'completed', '369 Willow Rd'),
                     (10, '2023-04-01', 499.99, 'completed', '741 Ash Dr'),
                     (11, '2023-04-05', 189.99, 'pending', '852 Oak St'),
                     (12, '2023-04-10', 649.99, 'completed', '963 Pine Ave'),
                     (13, '2023-04-15', 89.99, 'cancelled', '159 Maple Ln'),
                     (14, '2023-04-20', 299.99, 'completed', '357 Cedar St'),
                     (15, '2023-04-25', 199.99, 'pending', '486 Elm Dr'),
                     (16, '2023-05-01', 749.99, 'completed', '729 Birch Rd'),
                     (17, '2023-05-05', 129.99, 'completed', '813 Spruce Ave'),
                     (18, '2023-05-10', 399.99, 'pending', '924 Willow St'),
                     (19, '2023-05-15', 549.99, 'completed', '537 Pine Dr')]

    cursor.executemany(
        '''
    INSERT INTO orders (customer_id, order_date, total_amount, status, shipping_address)
    VALUES (?, ?, ?, ?, ?)
    ''', sample_orders)

    sample_order_items = [(1, 'Laptop', 1, 999.99), (1, 'Mouse', 2, 25.99),
                          (2, 'Keyboard', 1, 75.50), (3, 'Monitor', 1, 299.99),
                          (4, 'Headphones', 1, 89.99),
                          (5, 'Printer', 1, 199.50),
                          (6, 'External SSD', 1, 449.99),
                          (7, 'Webcam', 1, 129.99),
                          (8, 'Gaming Console', 1, 899.99),
                          (9, 'Wireless Router', 1, 159.99),
                          (10, 'Smart Speaker', 2, 139.99),
                          (11, 'Tablet', 1, 499.99),
                          (12, 'Phone Case', 3, 29.99),
                          (13, 'Power Bank', 2, 49.99),
                          (14, 'USB Hub', 1, 39.99),
                          (15, 'Wireless Earbuds', 1, 199.99),
                          (16, 'Graphics Card', 1, 749.99),
                          (17, 'RAM Kit', 2, 64.99),
                          (18, 'CPU Cooler', 1, 89.99),
                          (19, 'Mechanical Keyboard', 1, 149.99)]

    cursor.executemany(
        '''
    INSERT INTO order_items (order_id, product_name, quantity, unit_price)
    VALUES (?, ?, ?, ?)
    ''', sample_order_items)

    sample_interactions = [
        (1, 'complaint', 'Laptop arrived with screen damage', '2023-03-15',
         'resolved', 'Replacement laptop sent'),
        (2, 'inquiry', 'Question about Gaming Console warranty', '2023-03-16',
         'closed', 'Provided warranty details'),
        (3, 'feedback', 'Great experience with Mechanical Keyboard',
         '2023-03-17', 'closed', 'Thanked customer'),
        (4, 'complaint', 'Headphones not delivered on time', '2023-03-18',
         'resolved', '$10 store credit issued'),
        (5, 'inquiry', 'RAM compatibility question', '2023-03-19', 'closed',
         'Confirmed compatibility'),
        (6, 'feedback', 'Monitor arrived poorly packaged', '2023-03-20',
         'resolved', 'Updated packaging protocol'),
        (7, 'complaint', 'Received wrong USB Hub model', '2023-03-21',
         'resolved', 'Correct model shipped'),
        (8, 'inquiry', 'Return policy for Graphics Card', '2023-03-22',
         'closed', 'Explained 30-day policy'),
        (9, 'feedback', 'Excellent support for Power Bank issue', '2023-03-23',
         'closed', 'Shared with team'),
        (10, 'complaint', 'Wireless Earbuds missing charging case',
         '2023-03-24', 'resolved', 'Sent missing case')
    ]

    sample_status_history = [
        (1, 'processing', '2023-03-15 10:00:00', 'Order received'),
        (1, 'shipped', '2023-03-15 14:30:00', 'Package handed to carrier'),
        (2, 'processing', '2023-03-16 09:15:00', 'Payment verified'),
        (3, 'delivered', '2023-03-17 13:20:00', 'Successfully delivered'),
        (4, 'shipped', '2023-03-18 11:45:00', 'In transit'),
        (5, 'processing', '2023-03-19 16:00:00', 'Awaiting stock'),
        (6, 'cancelled', '2023-03-20 10:30:00',
         'Customer requested cancellation'),
        (7, 'delivered', '2023-03-21 12:00:00', 'Left at front door'),
        (8, 'shipped', '2023-03-22 15:45:00', 'Out for delivery'),
        (9, 'processing', '2023-03-23 09:00:00', 'Packaging items')
    ]

    sample_product_issues = [
        (1, 1, 'defect', 'Dead pixels on laptop screen', '2023-03-15',
         'Replaced unit'),
        (2, 2, 'warranty', 'Gaming console not powering on', '2023-03-16',
         'Repair completed'),
        (3, 3, 'usage_help', 'Need help with keyboard settings', '2023-03-17',
         'Provided guide'),
        (4, 4, 'defect', 'Headphone jack loose', '2023-03-18',
         'Replacement sent'),
        (5, 5, 'warranty', 'RAM compatibility issue', '2023-03-19',
         'Exchanged for compatible model'),
        (6, 6, 'usage_help', 'Monitor calibration help', '2023-03-20',
         'Remote assistance provided'),
        (7, 7, 'defect', 'USB ports not working', '2023-03-21',
         'Replacement sent'),
        (8, 8, 'warranty', 'GPU overheating', '2023-03-22',
         'Under investigation'),
        (9, 9, 'usage_help', 'Power bank not charging', '2023-03-23',
         'Resolved - user error'),
        (10, 10, 'defect', 'Earbud case broken', '2023-03-24',
         'Replacement sent')
    ]

    cursor.executemany(
        '''
    INSERT INTO customer_interactions (customer_id, interaction_type, description, created_at, status, resolution_notes)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_interactions)

    cursor.executemany(
        '''
    INSERT INTO order_status_history (order_id, status, status_date, notes)
    VALUES (?, ?, ?, ?)
    ''', sample_status_history)

    cursor.executemany(
        '''
    INSERT INTO product_issues (order_item_id, customer_id, issue_type, description, reported_date, resolution)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_product_issues)

    conn.commit()
    conn.close()


create_tables_with_sample_data()


def get_table_schemas():
    conn = sqlite3.connect('customer_service.db')
    cursor = conn.cursor()

    schemas = {}
    tables = [
        'customers', 'orders', 'order_items', 'customer_interactions',
        'order_status_history', 'product_issues'
    ]

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        schemas[table] = [{
            'column_name': col[1],
            'data_type': col[2],
            'is_nullable': not col[3],
            'is_primary_key': bool(col[5]),
        } for col in columns]

    conn.close()
    return f"""Available Table: {tables} \n Available Columns: {schemas}"""
    # return schemas


async def classify_customer_service_query(customer_service_query: str):

    class ClassifyTask(BaseModel):
        classification_label: Literal["customer_service", "order_service",
                                      "product_service", "payment_service",
                                      "shipping_service", "return_service",
                                      "other"]

    # print(f"Customer Service Query: {customer_service_query}")
    functions = [{
        "name": "classify_customer_service_query",
        "description":
        "Classify the customer service query into one of the following categories: customer_service, order_service, product_service, payment_service, shipping_service, return_service, other",
        "parameters": ClassifyTask.model_json_schema(),
    }]
    messages = [
        {
            "role":
            "system",
            "content":
            "You are a customer service agent. You are given a customer service query and you need to classify it into one of the following categories: customer_service, order_service, product_service, payment_service, shipping_service, return_service, other",
        },
        {
            "role": "user",
            "content": f"Customer Service Query: {customer_service_query}"
        },
    ]
    classification_response = await llm.function_call(
        messages=messages,
        functions=functions,
        function_call={"name": "classify_customer_service_query"},
        model_name=model_name,
        temperature=0.2)
    # print(classification_response)
    classification_response = classification_response[0].get("parameters")
    return classification_response.get("classification_label")


async def create_sql_query(query_description: str, extra_details: str):

    class CreateSQLQueryTask(BaseModel):
        sql_query: str = Field(
            description=
            "The SQL query to retrieve the relevant data from the database.")

    schema = get_table_schemas()
    messages = [
        {
            "role":
            "system",
            "content":
            "You are a customer service agent. You are given a customer service query and you need to create a SQL query to retrieve the relevant data from the database. You're also provided with the schema of the database and any extra details that might help you create the SQL query.",
        },
        {
            "role":
            "user",
            "content":
            f"Query: {query_description}\n\nSchema: {schema}\n\nExtra Details: {extra_details}",
        },
    ]
    functions = [{
        "name": "create_sql_query",
        "description":
        "Create a SQL query to retrieve the relevant data from the database.",
        "parameters": CreateSQLQueryTask.model_json_schema(),
    }]
    sql_query_response = await llm.function_call(messages=messages,
                                                 functions=functions,
                                                 model_name=model_name)
    sql_query_response = sql_query_response[0].get("parameters")
    return sql_query_response.get("sql_query")


def execute_sql_query(sql_query: str):
    conn = sqlite3.connect('customer_service.db')
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)

        # For SELECT queries, cursor.description will contain column info
        if cursor.description:
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            result = [dict(zip(columns, row)) for row in rows]
        # For INSERT queries, get the inserted row
        elif sql_query.strip().upper().startswith('INSERT'):
            last_id = cursor.lastrowid
            table_name = sql_query.split('INTO')[1].split('(')[0].strip()

            # Fetch the inserted row
            cursor.execute(f"SELECT * FROM {table_name} WHERE rowid = ?",
                           (last_id, ))
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()

            result = {
                "affected_rows": cursor.rowcount,
                "last_inserted_id": last_id,
                "inserted_row": dict(zip(columns, row)) if row else None
            }
        # For UPDATE/DELETE queries, return affected row count
        else:
            result = {"affected_rows": cursor.rowcount}

        conn.commit()
    except Exception as e:
        result = {"error": str(e)}
    finally:
        conn.close()

    return result


async def generate_response(customer_service_query: str, query_label: str,
                            query_result: List[Dict]):
    messages = [{
        "role":
        "system",
        "content":
        "You are a customer service agent. You are given a customer service query and you need to generate a response to the query. You're also provided with the query label and the query result.",
    }, {
        "role":
        "user",
        "content":
        f"Query: {customer_service_query}\n\nQuery Label: {query_label}\n\nQuery Result: {query_result}",
    }]
    print(f"\n\n")
    collected_response = ""
    async for chunk in llm.stream_complete(messages=messages,
                                           model_name=model_name,
                                           temperature=0.2):
        collected_response += chunk
        # print(chunk, end="", flush=True)
    print("\n\n")
    return collected_response
    # return await llm.complete(messages=messages, model_name=model_name)


class InputForClassifyCustomerServiceQuery(
        BaseModel):  # Agent -> ClassifyCustomerServiceTask
    customer_service_query: str = Field(
        description="The customer service query to classify.")


class InputForCreateSQLQuery(BaseModel):  # Agent -> SQL Query Agent
    query_description: str = Field(
        description="The description of the query to create.")
    extra_details: Union[str, None] = Field(
        description=
        "Any extra details that might help the SQL Query Agent create the query."
    )


class InputForExecuteSQLQuery(BaseModel):  # Agent -> SQL Query Execution Agent
    sql_query: str = Field(
        description=
        "The SQL query to execute. This query should be created using the create_sql_query function."
    )


class InputForGenerateResponse(BaseModel):  # Agent -> Response Generator
    customer_service_query: str = Field(
        description="The customer service query to generate a response to.")
    query_label: str = Field(
        description="The label of the query to generate a response to.")
    query_result: Union[List[Dict], ContextVariable] = Field(
        description="The result of the query to generate a response to.")


# Agents

# Ticket Classifier Agent

ticket_classifier_agent = Agent(
    name="ticket_classifier_agent",
    instruction=
    f"You are a ticket classifier agent. You are given a customer service query and you need to classify it into one of the following categories: customer_service, order_service, product_service, payment_service, shipping_service, return_service, other. You've access to the following tables and columns: {get_table_schemas()}",
    functions=[{
        "name":
        "classify_customer_service_query",
        "description":
        "Classify the customer service query into one of the following categories: customer_service, order_service, product_service, payment_service, shipping_service, return_service, other",
        "parameters":
        InputForClassifyCustomerServiceQuery.model_json_schema(),
        "_callable":
        classify_customer_service_query
    }])

# SQL Query Agent

sql_query_generator_agent = Agent(
    name="sql_query_generator_agent",
    instruction=
    "You are a SQL query generator agent. You are given a customer service query and you need to create a SQL query to retrieve the relevant data from the database.",
    functions=[{
        "name": "create_sql_query",
        "description":
        "Create a SQL query to retrieve the relevant data from the database.",
        "parameters": InputForCreateSQLQuery.model_json_schema(),
        "_callable": create_sql_query
    }])

# SQL Query Execution Agent

sql_query_execution_agent = Agent(
    name="sql_query_execution_agent",
    instruction=
    "You are a SQL query execution agent. You are given a SQL query and you need to execute it and return the result.",
    functions=[{
        "name": "execute_sql_query",
        "description": "Execute a SQL query and return the result.",
        "parameters": InputForExecuteSQLQuery.model_json_schema(),
        "_callable": execute_sql_query
    }])

# Response Generator Agent

response_generator_agent = Agent(
    name="response_generator_agent",
    instruction=
    "You are a response generator agent. You are given a customer service query, the query label, and the query result. You need to generate a response to the customer service query.",
    functions=[{
        "name": "generate_response",
        "description": "Generate a response to the customer service query.",
        "parameters": InputForGenerateResponse.model_json_schema(),
        "_callable": generate_response
    }])

swarm = [
    ticket_classifier_agent, sql_query_generator_agent,
    sql_query_execution_agent, response_generator_agent
]

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instruction=
    "You are an orchestrator agent. You are given a customer service query and you need to orchestrate the swarm to solve the customer service query. You can use response_generator_agent to generate a response to the customer service query and ask the customer to provide more details if needed. You always need to generate a response to the customer service query.",
    functions=[])

extra_details = {"customer_id": 1}


async def main():
    messages = []
    support_pipeline = Loka(orchestrator_agent=orchestrator_agent,
                            swarm=swarm,
                            llm=llm,
                            max_iterations=10)
    try:
        while True:
            user_input = input("User: ")
            messages.append({"role": "user", "content": user_input})
            messages[-1]["content"] += f"\nExtra Details: {extra_details}"

            async for chunk in support_pipeline.swarmloka(
                    model_name,
                    messages,
                    write_end_result=False,
                    llm_args={"temperature": 0.2}):

                # Parse the chunk as dictionary
                if isinstance(chunk, str):
                    try:
                        chunk_dict = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                else:
                    chunk_dict = chunk

            called_agents = support_pipeline.selected_agents
            print(f"\n\n{json.dumps(called_agents, indent=4)}\n\n")
            response_gen_agent = list(
                filter(lambda x: x.get("agent") == "response_generator_agent",
                       called_agents))
            if len(response_gen_agent) == 0:
                raise Exception(
                    "Error resolving the issue. Please try again later.")
            response_gen_agent = response_gen_agent[-1]
            response_gen_agent_function = response_gen_agent.get(
                "functions")[-1]
            result = response_gen_agent_function.get("result")
            print(f"\n\n{result}\n\n")
            messages.append({"role": "assistant", "content": result})

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    asyncio.run(main())
