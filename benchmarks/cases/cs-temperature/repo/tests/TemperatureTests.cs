using Weather;
using Xunit;

public class TemperatureTests
{
    [Fact]
    public void BoilingPointToCelsius() => Assert.Equal(100.0, Temperature.FahrenheitToCelsius(212), 3);

    [Fact]
    public void FreezingPointToCelsius() => Assert.Equal(0.0, Temperature.FahrenheitToCelsius(32), 3);

    [Fact]
    public void CelsiusToFahrenheit() => Assert.Equal(212.0, Temperature.CelsiusToFahrenheit(100), 3);

    [Fact]
    public void RoundTrip() =>
        Assert.Equal(37.0, Temperature.FahrenheitToCelsius(Temperature.CelsiusToFahrenheit(37)), 3);
}
