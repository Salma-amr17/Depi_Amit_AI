import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QMessageBox

# ====== BankAccount Class ======
class BankAccount:
    def __init__(self, balance=0):
        if balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            return "Deposit amount must be positive"
        self.balance += amount
        return f"Deposited {amount}. New balance: {self.balance}"

    def withdraw(self, amount):
        if amount <= 0:
            return "Withdrawal amount must be positive"
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        return f"Withdrew {amount}. New balance: {self.balance}"

    def check_balance(self):
        return f"Current balance: {self.balance}"


# ====== GUI Application ======
class BankApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.account = BankAccount(100)   # Initial balance of 100
        self.setWindowTitle("Bank Account System")
        self.setGeometry(300, 200, 400, 200)

        # Label
        self.label = QLabel("Enter Amount:", self)
        self.label.move(20, 20)

        # Input Field
        self.input_field = QLineEdit(self)
        self.input_field.move(120, 20)
        self.input_field.resize(200, 25)

        # Buttons
        self.deposit_btn = QPushButton("Deposit", self)
        self.deposit_btn.move(20, 70)
        self.deposit_btn.clicked.connect(self.deposit_action)

        self.withdraw_btn = QPushButton("Withdraw", self)
        self.withdraw_btn.move(120, 70)
        self.withdraw_btn.clicked.connect(self.withdraw_action)

        self.balance_btn = QPushButton("Check Balance", self)
        self.balance_btn.move(220, 70)
        self.balance_btn.clicked.connect(self.check_balance_action)

    def deposit_action(self):
        try:
            amount = float(self.input_field.text())
            result = self.account.deposit(amount)
            QMessageBox.information(self, "Result", result)
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number")

    def withdraw_action(self):
        try:
            amount = float(self.input_field.text())
            result = self.account.withdraw(amount)
            QMessageBox.information(self, "Result", result)
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number")

    def check_balance_action(self):
        result = self.account.check_balance()
        QMessageBox.information(self, "Balance", result)


# ====== Run App ======
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BankApp()
    window.show()
    sys.exit(app.exec_())