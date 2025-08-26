import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox

# ====== BankAccount Class ======
class BankAccount:
    def __init__(self, balance=0):
        if balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.balance = balance
        self.history = []  # to store transaction history

    def deposit(self, amount):
        if amount <= 0:
            return "Deposit amount must be positive"
        self.balance += amount
        self.history.append(("Deposit", amount, self.balance))
        return f"Deposited {amount}. New balance: {self.balance}"

    def withdraw(self, amount):
        if amount <= 0:
            return "Withdrawal amount must be positive"
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        self.history.append(("Withdraw", amount, self.balance))
        return f"Withdrew {amount}. New balance: {self.balance}"

    def get_balance(self):
        return self.balance

    def get_history(self):
        return self.history


# ====== GUI Application ======
class BankApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.account = BankAccount(100)   # Initial balance of 100
        self.setWindowTitle("Bank Account System")
        self.setGeometry(300, 200, 500, 400)

        # Label + Current Balance
        self.balance_label = QLabel(f"Current Balance: {self.account.get_balance()}", self)
        self.balance_label.move(20, 20)
        self.balance_label.resize(300, 25)

        # Label + Input Field
        self.label = QLabel("Enter Amount:", self)
        self.label.move(20, 60)
        self.input_field = QLineEdit(self)
        self.input_field.move(120, 60)
        self.input_field.resize(200, 25)

        # Buttons
        self.deposit_btn = QPushButton("Deposit", self)
        self.deposit_btn.move(20, 100)
        self.deposit_btn.clicked.connect(self.deposit_action)

        self.withdraw_btn = QPushButton("Withdraw", self)
        self.withdraw_btn.move(120, 100)
        self.withdraw_btn.clicked.connect(self.withdraw_action)

        # Table for transaction history
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Action", "Amount", "Balance"])
        self.table.setGeometry(20, 150, 450, 200)

    def deposit_action(self):
        try:
            amount = float(self.input_field.text())
            result = self.account.deposit(amount)
            self.update_ui()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number")

    def withdraw_action(self):
        try:
            amount = float(self.input_field.text())
            result = self.account.withdraw(amount)
            self.update_ui()
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number")

    def update_ui(self):
        # Update balance label
        self.balance_label.setText(f"Current Balance: {self.account.get_balance()}")
        # Update transaction history table
        history = self.account.get_history()
        self.table.setRowCount(len(history))
        for i, (action, amount, balance) in enumerate(history):
            self.table.setItem(i, 0, QTableWidgetItem(action))
            self.table.setItem(i, 1, QTableWidgetItem(str(amount)))
            self.table.setItem(i, 2, QTableWidgetItem(str(balance)))


# ====== Run App ======
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BankApp()
    window.show()
    sys.exit(app.exec_())