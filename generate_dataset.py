import csv
import random
from faker import Faker

fake = Faker()

NUM_RECORDS = 1000

# Categories and their ratios: suspicious (10%), normal (80%), borderline (10%)
CATEGORIES = ['suspicious', 'normal', 'borderline']
CATEGORY_WEIGHTS = [0.1, 0.8, 0.1]

# Probability of generating a completely random noise query (to add variability)
NOISE_PROB = 0.1

# Probability of mislabeling (simulate labeling errors)
MISLABEL_PROB = 0.07

def random_sql_noise():
    """Generate a random or malformed SQL string."""
    tokens = [
        "SELECT", "*", "FROM", "WHERE", "JOIN", "ON", "UPDATE", "INSERT",
        "DROP", "TABLE", "VALUES", fake.word(), fake.word(),
        "(", ")", "=", ">", "<", "'X'", ";"
    ]
    length = random.randint(3, 8)
    return " ".join(random.choices(tokens, k=length)) + ";"

def generate_query(category):
    """Generate a SQL query with a chance to insert noise."""
    if random.random() < NOISE_PROB:
        return random_sql_noise()
    
    if category == 'suspicious':
        table = random.choice([
            "Employee_Salaries", "Credit_Card_Info", "Customer_Payments",
            "User_Login_History", "Employee_Personal_Info", "Sensitive_Logs"
        ])
        if table == "Employee_Salaries":
            role = random.choice(["Manager", "Director", "Supervisor"])
            return f"SELECT salary, bonus FROM {table} WHERE role='{role}';"
        elif table == "Credit_Card_Info":
            return f"SELECT * FROM {table};"
        elif table == "Customer_Payments":
            amt = random.randint(5000, 10000)
            return f"SELECT * FROM {table} WHERE amount > {amt};"
        elif table == "User_Login_History":
            return f"SELECT * FROM {table} WHERE success=0;"
        elif table == "Employee_Personal_Info":
            return f"SELECT * FROM {table};"
        else:
            return f"SELECT * FROM {table};"
    elif category == 'normal':
        table = random.choice([
            "Employees", "Products", "Orders", "Customers",
            "Product_Reviews", "Inventory"
        ])
        if table == "Employees":
            return f"SELECT employee_id, first_name, last_name FROM {table};"
        elif table == "Products":
            return f"SELECT product_id, product_name, price FROM {table};"
        elif table == "Orders":
            return f"SELECT order_id, order_date FROM {table} WHERE status='Completed';"
        elif table == "Customers":
            return f"SELECT customer_id, name FROM {table};"
        elif table == "Product_Reviews":
            rating_threshold = random.choice([3, 4, 5])
            return f"SELECT review_id, product_id, rating, comment FROM {table} WHERE rating >= {rating_threshold};"
        else:
            return f"SELECT * FROM {table};"
    elif category == 'borderline':
        table = random.choice(["Customer_Payments", "Orders", "User_Activity"])
        if table == "Customer_Payments":
            amt = random.randint(2000, 3000)
            return f"SELECT * FROM {table} WHERE amount > {amt};"
        elif table == "Orders":
            return f"SELECT * FROM {table} WHERE status='Pending';"
        else:
            return f"SELECT * FROM {table};"
    return ""

def generate_columns(category):
    """Randomly select 3-5 column names from a pool and occasionally add a random word."""
    if category == 'suspicious':
        pool = [
            "salary", "bonus", "employee_id", "department", "card_number", "expiry",
            "cvv", "customer_id", "user_id", "timestamp", "ip_address", "name",
            "address", "phone", "social_security_number"
        ]
    elif category == 'normal':
        pool = [
            "employee_id", "first_name", "last_name", "product_id", "product_name",
            "order_id", "order_date", "customer_id", "price", "status", "email",
            "event_id", "event_name", "review_id", "rating", "comment"
        ]
    else:  # borderline
        pool = [
            "payment_id", "amount", "customer_id", "order_id", "status", "order_date",
            "user_id", "activity", "timestamp"
        ]
    selected = random.sample(pool, random.randint(3, 5))
    if random.random() < 0.15:
        selected.append(fake.word())
    random.shuffle(selected)
    return ", ".join(selected)

def generate_role(category):
    """Select a realistic role; sometimes use Faker for extra variability."""
    if category == 'suspicious':
        roles = ["HR Intern", "Trainee", "Data Entry Clerk", "Junior Analyst", "Intern"]
    elif category == 'normal':
        roles = [
            "CEO", "CTO", "CFO", "COO", "CMO", "HR Manager", "Sales Manager",
            "Marketing Specialist", "Customer Support Representative", "Software Engineer",
            "Data Scientist", "IT Specialist", "Finance Analyst", "Operations Manager",
            "Product Manager", "Business Analyst", "Project Manager", "Accountant",
            "Auditor", "Legal Counsel", "Procurement Manager", "Security Analyst",
            "Systems Administrator"
        ]
    else:
        roles = ["Team Lead", "Supervisor", "IT Analyst", "Order Processor", "Regional Manager"]
    
    if random.random() < 0.1:
        return fake.job()
    else:
        return random.choice(roles)

def generate_rows(category):
    """Return a random row count, with overlapping ranges to simulate ambiguity."""
    if category == 'suspicious':
        return random.randint(20, 500)
    elif category == 'normal':
        return random.randint(50, 1000)
    elif category == 'borderline':
        return random.randint(30, 600)
    return random.randint(50, 1000)

def get_label(category):
    """
    For binary classification with label smoothing:
      - suspicious -> 0.05 (unsafe)
      - normal -> 0.95 (safe)
      - borderline -> 0.70 (ambiguous, leaning safe)
    """
    if category == 'suspicious':
        return 0.05
    elif category == 'normal':
        return 0.95
    elif category == 'borderline':
        return 0.70
    return 0.95

def generate_record(category):
    """Generate one record with a query, result, role, and a smoothed label.
       Also, apply a small chance of mislabeling to simulate real-world errors.
    """
    query = generate_query(category)
    columns = generate_columns(category)
    user_role = generate_role(category)
    rows_returned = generate_rows(category)
    
    phrasing = [
        f"{rows_returned} rows returned; columns: {columns}.",
        f"Retrieved {rows_returned} rows with columns: {columns}",
        f"{rows_returned} rows found, including columns: {columns}"
    ]
    result = random.choice(phrasing)
    label = get_label(category)
    
    # Introduce mislabeling with a small probability.
    if random.random() < MISLABEL_PROB:
        label = 1 - label  # Flip the label.
    
    return {
        "query": query,
        "result": result,
        "role": user_role,
        "label": label
    }

def main():
    dataset = []
    for _ in range(NUM_RECORDS):
        category = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        record = generate_record(category)
        dataset.append(record)
    
    csv_filename = "synthetic_dataset.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["query", "result", "role", "label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in dataset:
            writer.writerow(record)
    
    print(f"Dataset generated with {len(dataset)} records and saved to '{csv_filename}'.")

if __name__ == "__main__":
    main()
