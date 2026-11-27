using Banking;
using Xunit;

public class AccountTests
{
    [Fact]
    public void OverdraftIsRefused()
    {
        var account = new Account(50m);
        Assert.Throws<InvalidOperationException>(() => account.Withdraw(80m));
        Assert.Equal(50m, account.Balance);
    }

    [Fact]
    public void CanWithdrawEntireBalance()
    {
        var account = new Account(50m);
        account.Withdraw(50m);
        Assert.Equal(0m, account.Balance);
    }

    [Fact]
    public void DepositAddsMoney()
    {
        var account = new Account(10m);
        account.Deposit(5m);
        Assert.Equal(15m, account.Balance);
    }
}
