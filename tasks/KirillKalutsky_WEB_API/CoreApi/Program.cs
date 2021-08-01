using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using System.IO;

namespace CoreApi
{
    public class Program
    {
        
        public static void Main(string[] args)
        {
            
            CreateHostBuilder(args).UseContentRoot(Directory.GetCurrentDirectory()).Build().Run();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureWebHostDefaults(webBuilder =>
                {
                    webBuilder.UseStartup<Startup>();
                });
    }
}
