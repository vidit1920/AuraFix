namespace Banking;

public class Account
{
    public decimal Balance { get; private set; }

    public Account(decimal openingBalance)
    {
        Balance = openingBalance;
    }

    public void Deposit(decimal amount)
    {
        if (amount <= 0) throw new ArgumentException("Deposit must be positive");
        Balance += amount;
    }

    public void Withdraw(decimal amount)
    {
        if (amount <= 0) throw new ArgumentException("Withdrawal must be positive");
        if (Balance < 0) throw new InvalidOperationException("Insufficient funds");
        Balance -= amount;
    }
}
