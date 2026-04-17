import { describe, expect, test } from "bun:test";
import { cn } from "../index";

describe("cn", () => {
  test("joins class names", () => {
    expect(cn("a", "b", "c")).toBe("a b c");
  });

  test("filters falsy values", () => {
    expect(cn("a", false, null, undefined, "b")).toBe("a b");
  });

  test("returns empty string for no args", () => {
    expect(cn()).toBe("");
  });
});
