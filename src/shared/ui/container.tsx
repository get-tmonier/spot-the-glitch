import { cn } from "@/shared/lib";

type ContainerProps = React.HTMLAttributes<HTMLDivElement>;

export function Container({ className, ...props }: ContainerProps) {
  return <div className={cn("mx-auto max-w-4xl px-4", className)} {...props} />;
}
